import glob
import os
import sys
from typing import Dict, Optional

os.chdir(os.path.dirname(os.path.abspath(__file__)))
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import argparse
import ast
import json
import logging
import random

import numpy as np
import pandas as pd
import torch
from routerl                import TrafficEnvironment
from tqdm                   import tqdm

from algorithms.simple_dqn  import DQN
from utils                  import clear_SUMO_files
from utils                  import print_agent_counts
from utils                  import start_runtime_tracking
from utils                  import finish_runtime_tracking

try:
    import wandb
except ImportError:
    wandb = None


def _load_wandb_secrets(config_path: str) -> Dict[str, str]:
    """Load W&B credentials/project defaults from json if it exists."""
    if not os.path.exists(config_path):
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _ensure_recorder_flush(env: TrafficEnvironment) -> None:
    """Wait for any pending async Recorder writes so episode files are ready."""
    pending = getattr(env, "pending_futures", None)
    if not pending:
        return
    for future in list(pending):
        future.result()
    env.pending_futures = []


def _log_new_episodes(
    wb_run: Optional["wandb.sdk.wandb_run.Run"],
    episodes_folder: str,
    last_logged: int,
    phase: str,
    env: TrafficEnvironment,
) -> int:
    """Log per-episode mean rewards/travel_times grouped by agent kind."""
    if wb_run is None:
        return last_logged

    _ensure_recorder_flush(env)

    ep_files = glob.glob(os.path.join(episodes_folder, "ep*.csv"))
    if not ep_files:
        return last_logged

    # Parse episode numbers and iterate in order
    ep_entries = []
    for ep_path in ep_files:
        try:
            ep_num = int(os.path.basename(ep_path).split("ep")[1].split(".csv")[0])
            ep_entries.append((ep_num, ep_path))
        except Exception:
            continue

    for ep_num, ep_path in sorted(ep_entries):
        if ep_num <= last_logged:
            continue
        df = pd.read_csv(ep_path)
        metrics = {
            "episode": ep_num,
            f"{phase}/reward/all": df["reward"].mean(),
            f"{phase}/travel_time/all": df["travel_time"].mean(),
        }
        for kind, group in df.groupby("kind"):
            kind_key = kind.lower()
            metrics[f"{phase}/reward/{kind_key}"] = group["reward"].mean()
            metrics[f"{phase}/travel_time/{kind_key}"] = group["travel_time"].mean()
        wb_run.log(metrics, step=ep_num)
        last_logged = ep_num

    return last_logged


# Main script to run the IQL experiment
if __name__ == "__main__":
    cl = " ".join(sys.argv)
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str, required=True)
    parser.add_argument('--env-conf', type=str, default="config1")
    parser.add_argument('--task-conf', type=str, required=True)
    parser.add_argument('--alg-conf', type=str, required=True)
    parser.add_argument('--net', type=str, required=True)
    parser.add_argument('--env-seed', type=int, default=42)
    parser.add_argument('--torch-seed', type=int, default=42)
    parser.add_argument('--wandb-config', type=str, default=os.path.join(repo_root, "wandb_config.json"))
    parser.add_argument('--no-wandb', action='store_true', help="Disable Weights & Biases logging.")
    args = parser.parse_args()
    ALGORITHM = "iql"
    exp_id = args.id
    alg_config = args.alg_conf
    env_config = args.env_conf
    task_config = args.task_conf
    network = args.net
    env_seed = args.env_seed
    torch_seed = args.torch_seed
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

    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(torch_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(env_seed)
    np.random.seed(env_seed)

    device = (
        torch.device(0)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print("Device is: ", device)
        
    # Parameter setting
    params = dict()
    alg_params = json.load(open(f"../config/algo_config/{ALGORITHM}/{alg_config}.json"))
    env_params = json.load(open(f"../config/env_config/{env_config}.json"))
    task_params = json.load(open(f"../config/task_config/{task_config}.json"))
    params.update(alg_params)
    params.update(env_params)
    params.update(task_params)
    del params["desc"], env_params, task_params

    # set params as variables in this script
    for key, value in params.items():
        globals()[key] = value

    
    custom_network_folder = f"../networks/{network}"
    phases = [1, human_learning_episodes, int(training_eps) + human_learning_episodes]
    phase_names = ["Human stabilization", "Mutation and AV learning", "Testing phase"]
    records_folder = f"../results/{exp_id}"
    plots_folder = f"../results/{exp_id}/plots"
    episodes_folder = os.path.join(records_folder, "episodes")
    runtime_tracker = start_runtime_tracking(records_folder, exp_id, __file__, alg_config, task_config, env_config)

    # Read origin-destinations
    od_file_path = os.path.join(custom_network_folder, f"od_{network}.txt")
    with open(od_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    data = ast.literal_eval(content)
    origins = data['origins']
    destinations = data['destinations']

    
    # Copy agents.csv from custom_network_folder to records_folder
    agents_csv_path = os.path.join(custom_network_folder, "agents.csv")
    num_agents = len(pd.read_csv(agents_csv_path))
    if os.path.exists(agents_csv_path):
        os.makedirs(records_folder, exist_ok=True)
        new_agents_csv_path = os.path.join(records_folder, "agents.csv")
        with open(agents_csv_path, 'r', encoding='utf-8') as f:
            content = f.read()
        with open(new_agents_csv_path, 'w', encoding='utf-8') as f:
            f.write(content)
        max_start_time = pd.read_csv(new_agents_csv_path)['start_time'].max()
    else:
        raise FileNotFoundError(f"Agents CSV file not found at {agents_csv_path}. Please check the network folder.")
            
    num_machines = int(num_agents * ratio_machines)
    total_episodes = human_learning_episodes + training_eps + test_eps
            
    # Dump exp config to records
    exp_config_path = os.path.join(records_folder, "exp_config.json")
    dump_config = params.copy()
    dump_config["network"] = network
    dump_config["env_seed"] = env_seed
    dump_config["torch_seed"] = torch_seed
    dump_config["env_config"] = env_config
    dump_config["task_config"] = task_config
    dump_config["alg_config"] = alg_config
    dump_config["script"] = os.path.abspath(__file__)
    dump_config["algorithm"] = ALGORITHM
    dump_config["num_agents"] = num_agents
    dump_config["num_machines"] = num_machines
    dump_config["phases"] = phases
    dump_config["phase_names"] = phase_names
    dump_config["command"] = cl
    with open(exp_config_path, 'w', encoding='utf-8') as f:
        json.dump(dump_config, f, indent=4)

    # Initialize Weights & Biases (optional)
    if not args.no_wandb:
        if wandb is None:
            raise ImportError("wandb is not installed. Please install wandb or use --no-wandb.")
        wb_secrets = _load_wandb_secrets(args.wandb_config)
        api_key = wb_secrets.get("api_key")
        if api_key:
            wandb.login(key=api_key, relogin=True)
        wb_kwargs = {
            "project": wb_secrets.get("project", "openurb"),
            "entity": wb_secrets.get("entity"),
            "name": exp_id,
            "config": dump_config,
        }
        # Drop None values for cleaner init
        wb_kwargs = {k: v for k, v in wb_kwargs.items() if v is not None}
        wb_run = wandb.init(**wb_kwargs)
        wb_run.log({"status": "started"}, step=0)

    
    # Initialize the environment
    env = TrafficEnvironment(
        seed = env_seed,
        create_agents = False,
        create_paths = True,
        save_detectors_info = False,
        agent_parameters = {
            "new_machines_after_mutation": num_machines, 
            "human_parameters" : {
                "model" : human_model
            },
            "machine_parameters" : {
                "behavior" : av_behavior,
                "observation_type" : observations
            }
        },
        environment_parameters = {
            "save_every" : save_every,
        },
        simulator_parameters = {
            "network_name" : network,
            "custom_network_folder" : custom_network_folder,
            "sumo_type" : "sumo",
            "simulation_timesteps" : max_start_time
        }, 
        plotter_parameters = {
            "phases" : phases,
            "phase_names" : phase_names,
            "smooth_by" : smooth_by,
            "plot_choices" : plot_choices,
            "records_folder" : records_folder,
            "plots_folder" : plots_folder
        },
        path_generation_parameters = {
            "origins" : origins,
            "destinations" : destinations,
            "number_of_paths" : number_of_paths,
            "beta" : path_gen_beta,
            "num_samples" : num_samples,
            "visualize_paths" : False
        } 
    )

    env.start()
    env.reset()
    print_agent_counts(env)


    ### Human learning phase ###
    pbar = tqdm(total=total_episodes, desc="Human learning")
    for episode in range(human_learning_episodes):
        env.step()
        pbar.update()
        last_logged_episode = _log_new_episodes(
            wb_run, episodes_folder, last_logged_episode, "human_learning", env
        )


    # Mutation
    env.mutation(disable_human_learning = not should_humans_adapt, mutation_start_percentile = -1)
    print_agent_counts(env)
    obs_size = env.observation_space(env.possible_agents[0]).shape[0]
    
    # Set policies for machine agents
    for idx in range(len(env.machine_agents)):
        env.machine_agents[idx].model = DQN(obs_size, env.machine_agents[idx].action_space_size, 
                                            device=device, eps_init=eps_init, eps_decay=eps_decay,
                                            buffer_size=buffer_size, batch_size=batch_size, lr=lr, 
                                            num_epochs=num_epochs, num_hidden=num_hidden, widths=widths)
    agent_lookup = {str(agent.id): agent for agent in env.machine_agents}
    

    ### Learning phase ###
    pbar.set_description("AV learning")
    os.makedirs(plots_folder, exist_ok=True)
    for episode in range(training_eps):
        env.reset()
        for agent_id in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            
            if termination or truncation:
                agent_lookup[agent_id].model.push(reward)
                if episode % update_every == 0:
                    agent_lookup[agent_id].model.learn()
                action = None
            else:
                action = agent_lookup[agent_id].model.act(observation)
                
            env.step(action)
            
        if episode % plot_every == 0:
            env.plot_results()
        pbar.update()
        last_logged_episode = _log_new_episodes(
            wb_run, episodes_folder, last_logged_episode, "training", env
        )
    
    
    ### Testing phase ###
    for agent in env.machine_agents:
        agent.model.epsilon = 0.0
        agent.model.q_network.eval()
        
    pbar.set_description("Testing")
    for episode in range(test_eps):
        env.reset()
        for agent_id in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            if termination or truncation:
                action = None
            else:
                action = agent_lookup[agent_id].model.act(observation)
            env.step(action)
        pbar.update()
        last_logged_episode = _log_new_episodes(
            wb_run, episodes_folder, last_logged_episode, "testing", env
        )
    
    # Finalize the experiment
    pbar.close()
    env.plot_results()
    losses_pd = pd.DataFrame([{"id": agent.id, "losses": agent.model.loss} for agent in env.machine_agents])
    losses_pd.to_csv(os.path.join(records_folder, "losses.csv"), index=False)
    env.stop_simulation()
    clear_SUMO_files(os.path.join(records_folder, "SUMO_output"), episodes_folder, remove_additional_files=True)
    finish_runtime_tracking(runtime_tracker)
    _ensure_recorder_flush(env)
    _log_new_episodes(wb_run, episodes_folder, last_logged_episode, "final", env)
    if wb_run is not None:
        wb_run.log({"status": "finished"}, step=last_logged_episode + 1)
        wb_run.finish()
