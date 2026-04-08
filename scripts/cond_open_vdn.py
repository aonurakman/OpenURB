"""
Run the VDN benchmark in OpenURB with conditioned switching.

VDN and QMIX share the same recurrent per-agent Q-network structure.
The difference is the mixing rule: QMIX learns a monotonic mixer, while VDN
uses a fixed additive decomposition.
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

from algorithms.vdn import VDN
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
        # Some environment wrappers expose observations as {"observation": ..., "action_mask": ...}.
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


# Main script to run the conditioned VDN experiment
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

    ALGORITHM = "vdn"
    EXP_TYPE = "cond_open"
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
            conditional=True,
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
    params = dict()
    alg_params = json.load(open(f"../config/algo_config/{ALGORITHM}/{alg_config}.json"))
    env_params = json.load(open(f"../config/env_config/{env_config}.json"))
    task_params = json.load(open(f"../config/task_config/{task_config}.json"))
    params.update(alg_params)
    params.update(env_params)
    params.update(task_params)
    params.setdefault("share_parameters", True)
    del params["desc"], env_params, task_params

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

    # Track switching events between human and AV groups (conditioned on travel times).
    shifts_path = os.path.join(records_folder, "shifts.csv")
    shifts_df = pl.DataFrame(
        {col: list() for col in ["episode", "shifted_humans", "shifted_avs", "machine_ratio", "tt_ratio"]},
        schema={
            "episode": pl.Int64,
            "shifted_humans": pl.String,
            "shifted_avs": pl.String,
            "machine_ratio": pl.Float64,
            "tt_ratio": pl.Float64,
        },
    )

    od_file_path = os.path.join(custom_network_folder, f"od_{network}.txt")
    with open(od_file_path, "r", encoding="utf-8") as f:
        content = f.read()
    data = ast.literal_eval(content)
    origins = data["origins"]
    destinations = data["destinations"]

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
    pbar = tqdm(total=total_episodes, desc="Human learning")
    for episode in range(human_learning_episodes):
        env.step()
        pbar.update()
        last_logged_episode = log_new_episodes(wb_run, episodes_folder, last_logged_episode, "human_learning", env)

    ######### Mutation ########
    human_agents_copy = {str(agent.id): copy.deepcopy(agent) for agent in env.human_agents}
    env.mutation(disable_human_learning=not should_humans_adapt, mutation_start_percentile=-1)
    machine_agents_copy = {str(agent.id): copy.copy(agent) for agent in env.machine_agents}
    print_agent_counts(env)

    obs_size = env.observation_space(env.possible_agents[0]).shape[0]
    action_space_size = env.action_space_size

    agent_id_list = sorted(str(agent.id) for agent in env.all_agents)
    agent_id_to_index = {agent_id: idx for idx, agent_id in enumerate(agent_id_list)}

    ######## Set policy for machine agents ########
    vdn = VDN(
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
        num_epochs=num_epochs,
        num_hidden=num_hidden,
        widths=widths,
        rnn_hidden_dim=rnn_hidden_dim,
        max_grad_norm=max_grad_norm,
        gamma=gamma,
        target_update_every=target_update_every,
        double_q=double_q,
        tau=tau,
        share_parameters=share_parameters,
        q_tot_clip=q_tot_clip,
        use_huber_loss=use_huber_loss,
        normalize_by_active=globals().get("normalize_by_active", True),
    )

    for agent in env.machine_agents:
        agent.model = vdn
    for agent_id, agent in machine_agents_copy.items():
        agent.model = vdn

    ###############################################
    ######## AV learning + Switching phase ########
    ###############################################
    human_tts = list()
    av_tts = list()

    pbar.set_description("AV learning")
    for episode in range(training_eps + dynamic_episodes):
        travel_times = list()
        env.reset()
        vdn.reset_episode()

        active_mask = np.zeros(len(agent_id_list), dtype=np.float32)
        for agent_id in env.possible_agents:
            active_mask[agent_id_to_index[agent_id]] = 1.0

        obs_batch = np.zeros((len(agent_id_list), obs_size), dtype=np.float32)
        actions_batch = np.zeros(len(agent_id_list), dtype=np.int64)
        rewards_batch = np.zeros(len(agent_id_list), dtype=np.float32)

        for agent_id in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            obs, action_mask = extract_action_mask(observation, info, action_space_size)

            if termination or truncation:
                idx = agent_id_to_index[agent_id]
                rewards_batch[idx] = float(reward)
                action = None
            else:
                action = vdn.act(obs, action_mask=action_mask, agent_index=agent_id_to_index[agent_id])
                idx = agent_id_to_index[agent_id]
                obs_batch[idx] = obs
                actions_batch[idx] = int(action)

            env.step(action)
            travel_times.extend(env.travel_times_list)

        if episode > training_eps:
            ep_av_tt = [entry["travel_time"] for entry in travel_times if entry.get("kind") == "AV"]
            ep_human_tt = [entry["travel_time"] for entry in travel_times if entry.get("kind") == "Human"]
            if ep_av_tt:
                av_tts.append(np.mean(ep_av_tt))
            if ep_human_tt:
                human_tts.append(np.mean(ep_human_tt))

        vdn.store_episode(obs_batch, actions_batch, rewards_batch, active_mask)
        if episode % update_every == 0:
            vdn.learn()

        ##################################
        ######## Dynamic switches ########
        ##################################
        if (episode > training_eps) and (episode % switch_interval == 0):
            shifted_humans, shifted_avs = list(), list()

            for human_id in human_agents_copy:
                if human_id not in env.possible_agents:
                    agent_to_copy = next((agent for agent in env.human_agents if str(agent.id) == human_id), None)
                    assert (
                        agent_to_copy is not None
                    ), f"Human agent {human_id} not found in both possible agents and human agents."
                    human_agents_copy[human_id] = copy.deepcopy(agent_to_copy)

            for machine_id in machine_agents_copy:
                if machine_id in env.possible_agents:
                    agent_to_copy = next((agent for agent in env.machine_agents if str(agent.id) == machine_id), None)
                    assert (
                        agent_to_copy is not None
                    ), f"AV agent {machine_id} found in possible agents but not in machine agents."
                    machine_agents_copy[machine_id] = copy.copy(agent_to_copy)
                    machine_agents_copy[machine_id].model = vdn

            known_machines = set(machine_agents_copy.keys())
            tt_ratio = 1.0
            if (len(human_tts) > 0) and (len(av_tts) > 0):
                tt_ratio = float(np.mean(human_tts) / np.mean(av_tts))
            tt_ratio_denom = max(tt_ratio, 1e-6)
            cond_switch_prob_humans = min(1.0, max(0.0, switch_prob_humans * float(tt_ratio)))
            cond_switch_prob_machines = min(1.0, max(0.0, switch_prob_machines / float(tt_ratio_denom)))

            for human in env.human_agents[:]:
                if random.random() <= cond_switch_prob_humans:
                    env.human_agents.remove(human)
                    env.all_agents.remove(human)

                    human_id = str(human.id)
                    if human_id in known_machines:
                        new_av = copy.copy(machine_agents_copy[human_id])
                        new_av.model = vdn
                    else:
                        new_av = MachineAgent(
                            human.id,
                            human.start_time,
                            human.origin,
                            human.destination,
                            env.agent_params[kc.MACHINE_PARAMETERS],
                            env.action_space_size,
                        )
                        new_av.model = vdn

                    env.machine_agents.append(new_av)
                    shifted_humans.append(str(human.id))

            for machine in env.machine_agents[:]:
                if (str(machine.id) not in shifted_humans) and (random.random() <= cond_switch_prob_machines):
                    env.machine_agents.remove(machine)
                    env.all_agents.remove(machine)

                    new_human = copy.deepcopy(human_agents_copy[str(machine.id)])
                    env.human_agents.append(new_human)

                    shifted_avs.append(str(machine.id))

            env.all_agents = env.machine_agents + env.human_agents
            env._initialize_machine_agents()

            # Reset the travel-time windows after the team composition changes.
            human_tts = list()
            av_tts = list()

            shifted_humans = " ".join(shifted_humans) if shifted_humans else "None"
            shifted_avs = " ".join(shifted_avs) if shifted_avs else "None"
            shifts_df.extend(
                pl.DataFrame(
                    {
                        "episode": [episode],
                        "shifted_humans": [shifted_humans],
                        "shifted_avs": [shifted_avs],
                        "machine_ratio": [len(env.machine_agents) / len(env.all_agents)],
                        "tt_ratio": [tt_ratio],
                    }
                )
            )
            shifts_df.write_csv(shifts_path)

        if episode % plot_every == 0:
            env.plot_results()
        pbar.update()
        phase_label = "training" if episode < training_eps else "dynamic"
        last_logged_episode = log_new_episodes(wb_run, episodes_folder, last_logged_episode, phase_label, env)

    ###############################
    ######## Testing phase ########
    ###############################
    # Evaluate with a fixed low-noise Boltzmann policy.
    vdn.temperature = vdn.temp_min
    vdn.set_eval_mode()

    pbar.set_description("Testing")
    for episode in range(test_eps):
        global_episode = training_eps + dynamic_episodes + episode
        travel_times = list()
        env.reset()
        vdn.reset_episode()
        for agent_id in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            obs, action_mask = extract_action_mask(observation, info, action_space_size)
            if termination or truncation:
                action = None
            else:
                action = vdn.act(obs, action_mask=action_mask, agent_index=agent_id_to_index[agent_id])
            env.step(action)
            travel_times.extend(env.travel_times_list)
        if global_episode > training_eps:
            ep_av_tt = [entry["travel_time"] for entry in travel_times if entry.get("kind") == "AV"]
            ep_human_tt = [entry["travel_time"] for entry in travel_times if entry.get("kind") == "Human"]
            if ep_av_tt:
                av_tts.append(np.mean(ep_av_tt))
            if ep_human_tt:
                human_tts.append(np.mean(ep_human_tt))

        if (global_episode > training_eps) and (global_episode % switch_interval == 0):
            shifted_humans, shifted_avs = list(), list()

            for human_id in human_agents_copy:
                if human_id not in env.possible_agents:
                    agent_to_copy = next((agent for agent in env.human_agents if str(agent.id) == human_id), None)
                    assert (
                        agent_to_copy is not None
                    ), f"Human agent {human_id} not found in both possible agents and human agents."
                    human_agents_copy[human_id] = copy.deepcopy(agent_to_copy)

            for machine_id in machine_agents_copy:
                if machine_id in env.possible_agents:
                    agent_to_copy = next((agent for agent in env.machine_agents if str(agent.id) == machine_id), None)
                    assert (
                        agent_to_copy is not None
                    ), f"AV agent {machine_id} found in possible agents but not in machine agents."
                    machine_agents_copy[machine_id] = copy.copy(agent_to_copy)
                    machine_agents_copy[machine_id].model = vdn

            known_machines = set(machine_agents_copy.keys())
            tt_ratio = 1.0
            if (len(human_tts) > 0) and (len(av_tts) > 0):
                tt_ratio = float(np.mean(human_tts) / np.mean(av_tts))
            tt_ratio_denom = max(tt_ratio, 1e-6)
            cond_switch_prob_humans = min(1.0, max(0.0, switch_prob_humans * float(tt_ratio)))
            cond_switch_prob_machines = min(1.0, max(0.0, switch_prob_machines / float(tt_ratio_denom)))

            for human in env.human_agents[:]:
                if random.random() <= cond_switch_prob_humans:
                    env.human_agents.remove(human)
                    env.all_agents.remove(human)

                    human_id = str(human.id)
                    if human_id in known_machines:
                        new_av = copy.copy(machine_agents_copy[human_id])
                        new_av.model = vdn
                    else:
                        new_av = MachineAgent(
                            human.id,
                            human.start_time,
                            human.origin,
                            human.destination,
                            env.agent_params[kc.MACHINE_PARAMETERS],
                            env.action_space_size,
                        )
                        new_av.model = vdn

                    env.machine_agents.append(new_av)
                    shifted_humans.append(str(human.id))

            for machine in env.machine_agents[:]:
                if (str(machine.id) not in shifted_humans) and (random.random() <= cond_switch_prob_machines):
                    env.machine_agents.remove(machine)
                    env.all_agents.remove(machine)

                    new_human = copy.deepcopy(human_agents_copy[str(machine.id)])
                    env.human_agents.append(new_human)

                    shifted_avs.append(str(machine.id))

            env.all_agents = env.machine_agents + env.human_agents
            env._initialize_machine_agents()

            # Reset the travel-time windows after the team composition changes.
            human_tts = list()
            av_tts = list()

            shifted_humans = " ".join(shifted_humans) if shifted_humans else "None"
            shifted_avs = " ".join(shifted_avs) if shifted_avs else "None"
            shifts_df.extend(
                pl.DataFrame(
                    {
                        "episode": [global_episode],
                        "shifted_humans": [shifted_humans],
                        "shifted_avs": [shifted_avs],
                        "machine_ratio": [len(env.machine_agents) / len(env.all_agents)],
                        "tt_ratio": [tt_ratio],
                    }
                )
            )
            shifts_df.write_csv(shifts_path)

        pbar.update()
        last_logged_episode = log_new_episodes(wb_run, episodes_folder, last_logged_episode, "testing", env)

    pbar.close()
    env.plot_results()
    losses_pd = pd.DataFrame([{"id": "vdn", "losses": vdn.loss}])
    losses_pd.to_csv(os.path.join(records_folder, "losses.csv"), index=False)
    save_mean_loss_plot(records_folder, {row["id"]: row["losses"] for row in losses_pd.to_dict("records")})
    final_model_path = os.path.join(records_folder, "final_model.pt")
    agent_state = vdn.agent_net.state_dict() if vdn.share_parameters else vdn.agent_nets.state_dict()
    target_agent_state = vdn.target_agent_net.state_dict() if vdn.share_parameters else vdn.target_agent_nets.state_dict()
    torch.save(
        {
            "algorithm": "vdn",
            "share_parameters": vdn.share_parameters,
            "agent_state_dict": agent_state,
            "target_agent_state_dict": target_agent_state,
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
