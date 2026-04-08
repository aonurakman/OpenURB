"""
This script runs PIMAC in the OpenURB traffic environment with open (predefined) switching.

PIMAC keeps MAPPO PPO semantics and adds teacher-student coordination learning:
- shared decentralized recurrent actor for AV decisions,
- tokenized centralized teacher-critic over the active team set,
- uncertainty-gated FiLM conditioning in the actor policy path,
- PPO clipping, entropy regularization, and team-level GAE(lambda),
- student context distillation from teacher context targets.
"""

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import argparse
import ast
import copy
import json
import logging
import random

import numpy as np
import pandas as pd
import polars as pl
import torch
from routerl import Keychain as kc
from routerl import TrafficEnvironment
from routerl import MachineAgent
from tqdm import tqdm

from algorithms.pimac import PIMAC
from utils import clear_SUMO_files
from utils import ensure_recorder_flush
from utils import finish_wandb_run
from utils import init_wandb_run
from utils import log_new_episodes
from utils import print_agent_counts
from utils import generate_exp_id
from utils import run_metrics
from utils import start_runtime_tracking
from utils import finish_runtime_tracking
from utils import save_mean_loss_plot


def coerce_observation(observation):
    """
    Convert environment observations into the actor-ready float32 vector format.

    RouteRL may return direct vectors or wrappers containing "observation"/"obs".
    """
    obs = observation
    if isinstance(observation, dict):
        obs = observation.get("observation", observation.get("obs", observation))
    return np.asarray(obs, dtype=np.float32)


# Main script to run the PIMAC experiment.
if __name__ == "__main__":
    cl = " ".join(sys.argv)
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=str, default=None, help="Experiment ID (auto-generated if omitted).")
    parser.add_argument("--env-conf", type=str, default="config1")
    parser.add_argument("--task-conf", type=str, required=True)
    parser.add_argument("--alg-conf", type=str, required=True)
    parser.add_argument("--net", type=str, required=True)
    parser.add_argument("--env-seed", type=int, default=42)
    parser.add_argument("--torch-seed", type=int, default=42)
    parser.add_argument("--wandb-config", type=str, default=os.path.join(repo_root, "wandb_config.json"))
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging.")
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip intermediate/final environment plots to speed up long tuning runs.",
    )
    args = parser.parse_args()

    ALGORITHM = "pimac"
    EXP_TYPE = "open"
    exp_id = args.id
    alg_config = args.alg_conf
    env_config = args.env_conf
    task_config = args.task_conf
    network = args.net
    env_seed = args.env_seed
    torch_seed = args.torch_seed
    if not exp_id:
        exp_id = generate_exp_id(
            ALGORITHM,
            network,
            alg_config,
            env_config,
            task_config,
            env_seed,
            torch_seed,
            conditional=False,
            results_root=os.path.join(repo_root, "results"),
        )
        print(f"No --id provided; generated experiment ID: {exp_id}")

    wb_run = None
    last_logged_episode = 0
    print("### STARTING EXPERIMENT ###")
    print(f"Algorithm: {ALGORITHM.upper()}")
    print(f"Experiment ID: {exp_id}")
    print(f"Network: {network}")
    print(f"Environment seed: {env_seed}")
    print(f"Algorithm config: {alg_config}")
    print(f"Environment config: {env_config}")
    print(f"Task config: {task_config}")

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(torch_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(env_seed)
    np.random.seed(env_seed)

    device = torch.device(0) if torch.cuda.is_available() else torch.device("cpu")
    print("Device is: ", device)

    ###################################
    ######## Parameter setting ########
    ###################################
    # Merge algorithm, environment, and task configs into a single params dict.
    params = dict()
    alg_params = json.load(open(f"../config/algo_config/{ALGORITHM}/{alg_config}.json"))
    env_params = json.load(open(f"../config/env_config/{env_config}.json"))
    task_params = json.load(open(f"../config/task_config/{task_config}.json"))
    params.update(alg_params)
    params.update(env_params)
    params.update(task_params)
    del params["desc"], env_params, task_params

    # Expose config values as local variables for consistency with other scripts.
    for key, value in params.items():
        globals()[key] = value

    custom_network_folder = f"../networks/{network}"
    HUMAN_LEARNING_START = 1
    AV_TRAINING_START = human_learning_episodes
    EXCHANGES_START = AV_TRAINING_START + training_eps
    TESTING_START = EXCHANGES_START + dynamic_episodes
    phases = [HUMAN_LEARNING_START, AV_TRAINING_START, EXCHANGES_START, TESTING_START]
    phase_names = ["Human stabilization", "Mutation and AV stabilization", "Dynamic switches", "Testing phase"]
    records_folder = f"../results/{exp_id}"
    plots_folder = f"../results/{exp_id}/plots"
    episodes_folder = os.path.join(records_folder, "episodes")
    os.makedirs(plots_folder, exist_ok=True)
    runtime_tracker = start_runtime_tracking(records_folder, exp_id, __file__, alg_config, task_config, env_config)

    # Track switching events between human and AV groups.
    shifts_path = os.path.join(records_folder, "shifts.csv")
    shifts_df = pl.DataFrame(
        {col: list() for col in ["episode", "shifted_humans", "shifted_avs", "machine_ratio"]},
        schema={
            "episode": pl.Int64,
            "shifted_humans": pl.String,
            "shifted_avs": pl.String,
            "machine_ratio": pl.Float64,
        },
    )

    # Read origin-destination pairs for path generation.
    od_file_path = os.path.join(custom_network_folder, f"od_{network}.txt")
    with open(od_file_path, "r", encoding="utf-8") as f:
        content = f.read()
    data = ast.literal_eval(content)
    origins = data["origins"]
    destinations = data["destinations"]

    # Copy the demand file into the experiment records for reproducibility.
    agents_csv_path = os.path.join(custom_network_folder, "agents.csv")
    num_agents = len(pd.read_csv(agents_csv_path))
    if os.path.exists(agents_csv_path):
        os.makedirs(records_folder, exist_ok=True)
        new_agents_csv_path = os.path.join(records_folder, "agents.csv")
        with open(agents_csv_path, "r", encoding="utf-8") as f:
            content = f.read()
        with open(new_agents_csv_path, "w", encoding="utf-8") as f:
            f.write(content)
        max_start_time = pd.read_csv(new_agents_csv_path)["start_time"].max()
    else:
        raise FileNotFoundError(
            f"Agents CSV file not found at {agents_csv_path}. Please check the network folder."
        )

    num_machines = int(num_agents * ratio_machines)
    total_episodes = human_learning_episodes + training_eps + dynamic_episodes + test_eps

    ######## Dump exp config to records ########
    # Store the full experiment configuration to replay runs later.
    exp_config_path = os.path.join(records_folder, "exp_config.json")
    dump_config = params.copy()
    dump_config["exp_type"] = EXP_TYPE
    dump_config["script"] = os.path.abspath(__file__)
    dump_config["network"] = network
    dump_config["env_seed"] = env_seed
    dump_config["torch_seed"] = torch_seed
    dump_config["env_config"] = env_config
    dump_config["task_config"] = task_config
    dump_config["alg_config"] = alg_config
    dump_config["algorithm"] = ALGORITHM
    dump_config["num_agents"] = num_agents
    dump_config["num_machines"] = num_machines
    dump_config["phases"] = phases
    dump_config["phase_names"] = phase_names
    dump_config["command"] = cl
    with open(exp_config_path, "w", encoding="utf-8") as f:
        json.dump(dump_config, f, indent=4)

    wb_run = init_wandb_run(args.wandb_config, exp_id, dump_config, args.no_wandb)

    ######## Initialize the environment ########
    # Environment is AEC: agents act one-at-a-time with shared simulator state.
    env = TrafficEnvironment(
        seed=env_seed,
        create_agents=False,
        create_paths=True,
        save_detectors_info=False,
        agent_parameters={
            "new_machines_after_mutation": num_machines,
            "human_parameters": {"model": human_model},
            "machine_parameters": {"behavior": av_behavior, "observation_type": observations},
        },
        environment_parameters={"save_every": save_every},
        simulator_parameters={
            "network_name": network,
            "custom_network_folder": custom_network_folder,
            "sumo_type": "sumo",
            "simulation_timesteps": max_start_time,
        },
        plotter_parameters={
            "phases": phases,
            "phase_names": phase_names,
            "smooth_by": smooth_by,
            "plot_choices": plot_choices,
            "records_folder": records_folder,
            "plots_folder": plots_folder,
        },
        path_generation_parameters={
            "origins": origins,
            "destinations": destinations,
            "number_of_paths": number_of_paths,
            "beta": path_gen_beta,
            "num_samples": num_samples,
            "path_gen_workers" : path_gen_workers,
            "visualize_paths": False,
        },
    )

    env.start()
    env.reset()
    print_agent_counts(env)

    ######################################
    ######## Human learning phase ########
    ######################################
    # Humans adapt for a fixed number of episodes before AVs appear.
    pbar = tqdm(total=total_episodes, desc="Human learning")
    for episode in range(human_learning_episodes):
        env.step()
        pbar.update()
        last_logged_episode = log_new_episodes(wb_run, episodes_folder, last_logged_episode, "human_learning", env)

    ######### Mutation ########
    # We make object copies, so if agents switch back they resume where they left off.
    human_agents_copy = {str(agent.id): copy.deepcopy(agent) for agent in env.human_agents}
    env.mutation(disable_human_learning=not should_humans_adapt, mutation_start_percentile=-1)
    machine_agents_copy = {str(agent.id): copy.copy(agent) for agent in env.machine_agents}
    print_agent_counts(env)

    obs_size = env.observation_space(env.possible_agents[0]).shape[0]
    action_space_size = env.action_space_size

    # Fix an ordering over the full population so we can build fixed-size joint buffers.
    # Agents who are currently humans are masked out by `active_mask`.
    agent_id_list = sorted(str(agent.id) for agent in env.all_agents)
    agent_id_to_index = {agent_id: idx for idx, agent_id in enumerate(agent_id_list)}

    ######## Set policy for machine agents ########
    # PIMAC learner shared by all AVs (decentralized actor + teacher-critic + distillation).
    pimac = PIMAC(
        obs_size,
        action_space_size,
        device=device,
        buffer_size=buffer_size,
        batch_size=batch_size,
        lr=lr,
        num_epochs=num_epochs,
        num_hidden=num_hidden,
        widths=widths,
        rnn_hidden_dim=rnn_hidden_dim,
        clip_eps=clip_eps,
        gae_lambda=gae_lambda,
        normalize_advantage=normalize_advantage,
        entropy_coef=entropy_coef,
        value_coef=value_coef,
        max_grad_norm=max_grad_norm,
        gamma=gamma,
        critic_hidden_sizes=critic_hidden_sizes,
        set_embed_dim=set_embed_dim,
        set_encoder_hidden_sizes=set_encoder_hidden_sizes,
        include_team_size_feature=include_team_size_feature,
        num_tokens=num_tokens,
        distill_weight=distill_weight,
        teacher_ema_tau=teacher_ema_tau,
        hypernet_rank=hypernet_rank,
        hypernet_hidden_sizes=hypernet_hidden_sizes,
        hypernet_delta_init_scale=hypernet_delta_init_scale,
        hypernet_l2_coef=hypernet_l2_coef,
        ctx_logvar_min=ctx_logvar_min,
        ctx_logvar_max=ctx_logvar_max,
    )

    for agent in env.machine_agents:
        # Parameter sharing: each AV points to the same learner.
        agent.model = pimac
    for agent_id, agent in machine_agents_copy.items():
        # Ensure stored AV copies keep using the same learner after switches.
        agent.model = pimac

    ###############################################
    ######## AV learning + Switching phase ########
    ###############################################
    # Training data model:
    # - Environment interaction is AEC turn-taking, one action per AV per day via `env.agent_iter()`.
    # - Learning treats the whole day as one joint decision problem over the current AV group:
    #     (obs_i, action_i) for each AV i  ->  rewards_i at end of day.
    #
    # We therefore store one joint transition per day into PIMAC's on-policy buffer.
    pbar.set_description("AV learning")
    for episode in range(training_eps + dynamic_episodes):
        env.reset()
        pimac.reset_episode()

        # Active mask: which agents are AVs *today* (not "whose turn it is").
        active_mask = np.zeros(len(agent_id_list), dtype=np.float32)
        for agent_id in env.possible_agents:
            active_mask[agent_id_to_index[agent_id]] = 1.0

        # Joint buffers (fixed-size over the full population).
        obs_batch = np.zeros((len(agent_id_list), obs_size), dtype=np.float32)
        actions_batch = np.zeros(len(agent_id_list), dtype=np.int64)
        rewards_batch = np.zeros(len(agent_id_list), dtype=np.float32)

        for agent_id in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            obs = coerce_observation(observation)

            if termination or truncation:
                # AEC: rewards are assigned when the day finishes. During the terminal
                # Some AEC environments yield terminal turns so reward bookkeeping can complete cleanly.
                idx = agent_id_to_index[agent_id]
                rewards_batch[idx] = float(reward)
                action = None
            else:
                # Choose an action for the current agent only (AEC semantics).
                action = pimac.act(obs, agent_index=agent_id_to_index[agent_id])
                idx = agent_id_to_index[agent_id]
                obs_batch[idx] = obs
                actions_batch[idx] = int(action)

            env.step(action)

        # Store one joint transition for the day.
        pimac.store_episode(obs_batch, actions_batch, rewards_batch, active_mask)
        if episode % update_every == 0:
            pimac.learn()

        ##################################
        ######## Dynamic switches ########
        ##################################
        if (episode > training_eps) and (episode % switch_interval == 0):
            shifted_humans, shifted_avs = list(), list()

            for human_id in human_agents_copy:
                if human_id not in env.possible_agents:
                    # Refresh cached human states for agents that learned as humans.
                    agent_to_copy = next((agent for agent in env.human_agents if str(agent.id) == human_id), None)
                    assert (
                        agent_to_copy is not None
                    ), f"Human agent {human_id} not found in both possible agents and human agents."
                    human_agents_copy[human_id] = copy.deepcopy(agent_to_copy)

            for machine_id in machine_agents_copy:
                if machine_id in env.possible_agents:
                    # Refresh cached AV states for agents that learned as machines.
                    agent_to_copy = next((agent for agent in env.machine_agents if str(agent.id) == machine_id), None)
                    assert (
                        agent_to_copy is not None
                    ), f"AV agent {machine_id} found in possible agents but not in machine agents."
                    machine_agents_copy[machine_id] = copy.copy(agent_to_copy)
                    machine_agents_copy[machine_id].model = pimac

            known_machines = set(machine_agents_copy.keys())

            for human in env.human_agents[:]:
                if random.random() <= switch_prob_humans:
                    # Convert a human to an AV (reuse prior AV state if available).
                    env.human_agents.remove(human)
                    env.all_agents.remove(human)

                    human_id = str(human.id)
                    if human_id in known_machines:
                        new_av = copy.copy(machine_agents_copy[human_id])
                        new_av.model = pimac
                    else:
                        new_av = MachineAgent(
                            human.id,
                            human.start_time,
                            human.origin,
                            human.destination,
                            env.agent_params[kc.MACHINE_PARAMETERS],
                            env.action_space_size,
                        )
                        new_av.model = pimac

                    env.machine_agents.append(new_av)
                    shifted_humans.append(str(human.id))

            for machine in env.machine_agents[:]:
                if (str(machine.id) not in shifted_humans) and (random.random() <= switch_prob_machines):
                    # Convert an AV back to a human and remove from the AV pool.
                    env.machine_agents.remove(machine)
                    env.all_agents.remove(machine)

                    new_human = copy.deepcopy(human_agents_copy[str(machine.id)])
                    env.human_agents.append(new_human)

                    shifted_avs.append(str(machine.id))

            # Rebuild internal env bookkeeping after group changes.
            env.all_agents = env.machine_agents + env.human_agents
            env._initialize_machine_agents()

            # Record switches
            shifted_humans = " ".join(shifted_humans) if shifted_humans else "None"
            shifted_avs = " ".join(shifted_avs) if shifted_avs else "None"
            shifts_df.extend(
                pl.DataFrame(
                    {
                        "episode": [episode],
                        "shifted_humans": [shifted_humans],
                        "shifted_avs": [shifted_avs],
                        "machine_ratio": [len(env.machine_agents) / len(env.all_agents)],
                    }
                )
            )
            shifts_df.write_csv(shifts_path)

        # Regularly make plots and update the progress.
        if (not args.skip_plots) and (episode % plot_every == 0):
            env.plot_results()
        pbar.update()
        phase_label = "training" if episode < training_eps else "dynamic"
        last_logged_episode = log_new_episodes(wb_run, episodes_folder, last_logged_episode, phase_label, env)

    ###############################
    ######## Testing phase ########
    ###############################
    # Keep policy sampling active during evaluation.
    pimac.set_eval_mode()

    pbar.set_description("Testing")
    for episode in range(test_eps):
        global_episode = training_eps + dynamic_episodes + episode
        env.reset()
        pimac.reset_episode()
        for agent_id in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            obs = coerce_observation(observation)
            if termination or truncation:
                action = None
            else:
                action = pimac.act(obs, agent_index=agent_id_to_index[agent_id])
            env.step(action)
        if (global_episode > training_eps) and (global_episode % switch_interval == 0):
            shifted_humans, shifted_avs = list(), list()

            for human_id in human_agents_copy:
                if human_id not in env.possible_agents:
                    # Refresh cached human states for agents that learned as humans.
                    agent_to_copy = next((agent for agent in env.human_agents if str(agent.id) == human_id), None)
                    assert (
                        agent_to_copy is not None
                    ), f"Human agent {human_id} not found in both possible agents and human agents."
                    human_agents_copy[human_id] = copy.deepcopy(agent_to_copy)

            for machine_id in machine_agents_copy:
                if machine_id in env.possible_agents:
                    # Refresh cached AV states for agents that learned as machines.
                    agent_to_copy = next((agent for agent in env.machine_agents if str(agent.id) == machine_id), None)
                    assert (
                        agent_to_copy is not None
                    ), f"AV agent {machine_id} found in possible agents but not in machine agents."
                    machine_agents_copy[machine_id] = copy.copy(agent_to_copy)
                    machine_agents_copy[machine_id].model = pimac

            known_machines = set(machine_agents_copy.keys())

            for human in env.human_agents[:]:
                if random.random() <= switch_prob_humans:
                    # Convert a human to an AV (reuse prior AV state if available).
                    env.human_agents.remove(human)
                    env.all_agents.remove(human)

                    human_id = str(human.id)
                    if human_id in known_machines:
                        new_av = copy.copy(machine_agents_copy[human_id])
                        new_av.model = pimac
                    else:
                        new_av = MachineAgent(
                            human.id,
                            human.start_time,
                            human.origin,
                            human.destination,
                            env.agent_params[kc.MACHINE_PARAMETERS],
                            env.action_space_size,
                        )
                        new_av.model = pimac

                    env.machine_agents.append(new_av)
                    shifted_humans.append(str(human.id))

            for machine in env.machine_agents[:]:
                if (str(machine.id) not in shifted_humans) and (random.random() <= switch_prob_machines):
                    # Convert an AV back to a human and remove from the AV pool.
                    env.machine_agents.remove(machine)
                    env.all_agents.remove(machine)

                    new_human = copy.deepcopy(human_agents_copy[str(machine.id)])
                    env.human_agents.append(new_human)

                    shifted_avs.append(str(machine.id))

            # Rebuild internal env bookkeeping after group changes.
            env.all_agents = env.machine_agents + env.human_agents
            env._initialize_machine_agents()

            # Record switches
            shifted_humans = " ".join(shifted_humans) if shifted_humans else "None"
            shifted_avs = " ".join(shifted_avs) if shifted_avs else "None"
            shifts_df.extend(
                pl.DataFrame(
                    {
                        "episode": [global_episode],
                        "shifted_humans": [shifted_humans],
                        "shifted_avs": [shifted_avs],
                        "machine_ratio": [len(env.machine_agents) / len(env.all_agents)],
                    }
                )
            )
            shifts_df.write_csv(shifts_path)

        pbar.update()
        last_logged_episode = log_new_episodes(wb_run, episodes_folder, last_logged_episode, "testing", env)

    # Finalize the experiment
    pbar.close()
    if not args.skip_plots:
        env.plot_results()
    losses_pd = pd.DataFrame([{"id": "pimac", "losses": pimac.loss}])
    losses_pd.to_csv(os.path.join(records_folder, "losses.csv"), index=False)
    with open(os.path.join(records_folder, "pimac_loss_history.json"), "w", encoding="utf-8") as f:
        json.dump(pimac.loss_history, f, indent=2)
    save_mean_loss_plot(records_folder, {row["id"]: row["losses"] for row in losses_pd.to_dict("records")})
    final_model_path = os.path.join(records_folder, "final_model.pt")
    torch.save(
        {
            "algorithm": "pimac",
            "actor_state_dict": pimac.actor_net.state_dict(),
            "critic_state_dict": pimac.critic.state_dict(),
        },
        final_model_path,
    )
    env.stop_simulation()
    clear_SUMO_files(os.path.join(records_folder, "SUMO_output"), episodes_folder, remove_additional_files=True)
    finish_runtime_tracking(runtime_tracker)
    ensure_recorder_flush(env)
    last_logged_episode = log_new_episodes(wb_run, episodes_folder, last_logged_episode, "final", env)
    finish_wandb_run(wb_run, last_logged_episode)
    run_metrics(exp_id, repo_root)
