"""
This script implements the PI-MAC algorithm for reinforcement learning in a traffic environment.
The experiment involves dynamic switching between human and autonomous vehicle (AV) agents with predefined transition probabilities.

PI-MAC vs VDN/QMIX in this repo:
- PI-MAC uses VDN-style mixing (sum/mean over active agents) and distills a permutation-invariant set teacher
  into per-agent FiLM conditioning for scalable team composition context.
- QMIX uses a learned mixing network that conditions on a centralized global state.
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


def extract_action_mask(observation, info, action_space_size):
    """Extract an action mask if present and return a clean observation array."""
    action_mask = None
    obs = observation
    if isinstance(observation, dict):
        # PettingZoo sometimes wraps observations as {"observation": ..., "action_mask": ...}.
        action_mask = observation.get("action_mask")
        obs = observation.get("observation", observation.get("obs", observation))

    if action_mask is None and isinstance(info, dict):
        # Some environments provide the mask via info instead.
        action_mask = info.get("action_mask")

    if action_mask is not None:
        # Force mask into a flat {0,1} array aligned to action_space_size.
        # Note: action masks apply to *actions of the currently-selected agent* (AEC),
        # not to "which agents are allowed to act".
        action_mask = np.asarray(action_mask, dtype=np.int8).reshape(-1)
        if action_mask.shape[0] < action_space_size:
            padded = np.zeros(action_space_size, dtype=np.int8)
            padded[: action_mask.shape[0]] = action_mask
            action_mask = padded
        elif action_mask.shape[0] > action_space_size:
            action_mask = action_mask[:action_space_size]

    obs = np.asarray(obs, dtype=np.float32)
    return obs, action_mask


# Main script to run the VDN experiment
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
    args = parser.parse_args()

    ALGORITHM = "pimac"
    VERSION = "2"
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
            version=VERSION
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
    params.setdefault("share_parameters", True)
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
    # PI-MAC learner shared by all AVs. Like VDN, PI-MAC does not require a centralized global state.
    pimac = PIMAC(
        obs_size,
        action_space_size,
        num_agents=len(agent_id_list),
        device=device,
        temp_init=temp_init,
        temp_decay=temp_decay,
        temp_min=temp_min,
        buffer_size=buffer_size,
        batch_size=batch_size,
        lr=lr,
        teacher_lr=globals().get("teacher_lr", lr),
        num_epochs=num_epochs,
        num_hidden=num_hidden,
        widths=widths,
        rnn_hidden_dim=rnn_hidden_dim,
        ctx_dim=globals().get("ctx_dim", 16),
        num_tokens=globals().get("num_tokens", 8),
        tok_dim=globals().get("tok_dim", globals().get("ctx_dim", 16)),
        teacher_emb_dim=globals().get("teacher_emb_dim", 128),
        teacher_hidden_sizes=globals().get("teacher_hidden_sizes", (128, 128)),
        teacher_attn_dim=globals().get("teacher_attn_dim", None),
        teacher_drop_prob=globals().get("teacher_drop_prob", 0.5),
        distill_weight=globals().get("distill_weight", 1.0),
        token_smooth_weight=globals().get("token_smooth_weight", 0.0),
        max_grad_norm=max_grad_norm,
        gamma=gamma,
        target_update_every=target_update_every,
        double_q=double_q,
        tau=tau,
        share_parameters=share_parameters,
        q_tot_clip=q_tot_clip,
        use_huber_loss=use_huber_loss,
        normalize_by_active=globals().get("normalize_by_active", True),
        obs_index_dim=globals().get("obs_index_dim", 3),
        obs_skip=globals().get("obs_skip", False),
        context_gate_reg=globals().get("context_gate_reg", 0.0),
        subteam_samples=globals().get("subteam_samples", 0),
        subteam_keep_prob=globals().get("subteam_keep_prob", 0.75),
        subteam_td_weight=globals().get("subteam_td_weight", 0.5),
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
    # We therefore store one joint transition per day into VDN's replay buffer.
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
            obs, action_mask = extract_action_mask(observation, info, action_space_size)

            if termination or truncation:
                # AEC: rewards are assigned when the day finishes. During the terminal
                # "dead steps", PettingZoo yields each agent again so we can read its reward.
                idx = agent_id_to_index[agent_id]
                rewards_batch[idx] = float(reward)
                action = None
            else:
                # Choose an action for the current agent only (AEC semantics).
                action = pimac.act(obs, action_mask=action_mask, agent_index=agent_id_to_index[agent_id])
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
        if episode % plot_every == 0:
            env.plot_results()
        pbar.update()
        phase_label = "training" if episode < training_eps else "dynamic"
        last_logged_episode = log_new_episodes(wb_run, episodes_folder, last_logged_episode, phase_label, env)

    ###############################
    ######## Testing phase ########
    ###############################
    # Freeze exploration and run deterministic policy for evaluation.
    pimac.temperature = 0.0
    pimac.set_eval_mode()

    pbar.set_description("Testing")
    for episode in range(test_eps):
        env.reset()
        pimac.reset_episode()
        for agent_id in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            obs, action_mask = extract_action_mask(observation, info, action_space_size)
            if termination or truncation:
                action = None
            else:
                # Action masking is still applied during evaluation.
                action = pimac.act(obs, action_mask=action_mask, agent_index=agent_id_to_index[agent_id])
            env.step(action)
        pbar.update()
        last_logged_episode = log_new_episodes(wb_run, episodes_folder, last_logged_episode, "testing", env)

    # Finalize the experiment
    pbar.close()
    env.plot_results()
    losses_pd = pd.DataFrame([{"id": "pimac", "losses": pimac.loss}])
    losses_pd.to_csv(os.path.join(records_folder, "losses.csv"), index=False)
    save_mean_loss_plot(records_folder, {row["id"]: row["losses"] for row in losses_pd.to_dict("records")})
    final_model_path = os.path.join(records_folder, "final_model.pt")
    agent_state = pimac.agent_net.state_dict() if pimac.share_parameters else pimac.agent_nets.state_dict()
    target_state = pimac.target_agent_net.state_dict() if pimac.share_parameters else pimac.target_agent_nets.state_dict()
    torch.save(
        {
            "algorithm": "pimac",
            "share_parameters": pimac.share_parameters,
            "agent_state_dict": agent_state,
            "target_agent_state_dict": target_state,
            "teacher_state_dict": pimac.teacher.state_dict(),
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
