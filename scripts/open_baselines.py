"""
This script is used to train AV agents using the baseline methods in a traffic simulation environment.
The experiment involves dynamic switching between human and autonomous vehicle (AV) agents with predefined transition probabilities.
Baseline methods can be found in the baseline_models/ directory.
"""

import os
import sys


os.chdir(os.path.dirname(os.path.abspath(__file__)))
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import argparse
import ast
import copy
import json
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from routerl import Keychain as kc
from routerl import TrafficEnvironment
from routerl import MachineAgent
from tqdm import tqdm

from utils import clear_SUMO_files
from utils import ensure_recorder_flush
from utils import finish_wandb_run
from utils import init_wandb_run
from utils import log_new_episodes
from utils import start_runtime_tracking
from utils import finish_runtime_tracking
from baseline_models import get_baseline

if __name__ == "__main__":
    cl = " ".join(sys.argv)
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str, required=True)
    parser.add_argument('--alg-conf', type=str, required=True)
    parser.add_argument('--env-conf', type=str, default="config1")
    parser.add_argument('--task-conf', type=str, required=True)
    parser.add_argument('--net', type=str, required=True)
    parser.add_argument('--env-seed', type=int, default=42)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--wandb-config', type=str, default=os.path.join(repo_root, "wandb_config.json"))
    parser.add_argument('--no-wandb', action='store_true', help="Disable Weights & Biases logging.")
    args = parser.parse_args()
    ALGORITHM = "baseline"
    EXP_TYPE = "open"
    exp_id = args.id
    alg_config = args.alg_conf
    env_config = args.env_conf
    task_config = args.task_conf
    network = args.net
    env_seed = args.env_seed
    baseline_model = args.model
    wb_run = None
    last_logged_episode = 0
    print("### STARTING EXPERIMENT ###")
    print(f"Experiment ID: {exp_id}")
    print(f"Network: {network}")
    print(f"Environment seed: {env_seed}")
    print(f"Algorithm config: {alg_config}")
    print(f"Environment config: {env_config}")
    print(f"Task config: {task_config}")
    print(f"Baseline model: {baseline_model}")

    # Check if baseline exists
    baseline_dir = Path(repo_root) / "baseline_models"
    available_models = {file.stem for file in baseline_dir.glob("*.py")}
    assert baseline_model in available_models, \
        f"Baseline model '{baseline_model}' not found in {baseline_dir}/. Available: {sorted(available_models)}"

    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    random.seed(env_seed)
    np.random.seed(env_seed)
        
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
    del alg_params, env_params, task_params

    # set params as variables in this script
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
    
    # To be used for tracking switches between groups
    shifts_path = os.path.join(records_folder, "shifts.csv")
    shifts_df = pl.DataFrame(
        {col : list() for col in ["episode", "shifted_humans", "shifted_avs", "machine_ratio"]},
        schema={"episode": pl.Int64, "shifted_humans": pl.String, "shifted_avs": pl.String, "machine_ratio": pl.Float64}
        )

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
    total_episodes = human_learning_episodes + training_eps + dynamic_episodes + test_eps
            
    ######## Dump exp config to records ########
    exp_config_path = os.path.join(records_folder, "exp_config.json")
    dump_config = params.copy()
    dump_config["exp_type"] = EXP_TYPE
    dump_config["script"] = os.path.abspath(__file__)
    dump_config["network"] = network
    dump_config["env_seed"] = env_seed
    dump_config["env_config"] = env_config
    dump_config["task_config"] = task_config
    dump_config["alg_config"] = alg_config
    dump_config["baseline_model"] = baseline_model
    dump_config["algorithm"] = ALGORITHM
    dump_config["num_agents"] = num_agents
    dump_config["num_machines"] = num_machines
    dump_config["phases"] = phases
    dump_config["phase_names"] = phase_names
    dump_config["command"] = cl
    with open(exp_config_path, 'w', encoding='utf-8') as f:
        json.dump(dump_config, f, indent=4)

    wb_run = init_wandb_run(args.wandb_config, exp_id, dump_config, args.no_wandb)

    # Initiate the traffic environment
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
            "machine_parameters" :{
                "behavior" : av_behavior,
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

    print(f"""
    Agents in the traffic:
    - Total agents           : {len(env.all_agents)}
    - Human agents           : {len(env.human_agents)}
    - AV agents              : {len(env.machine_agents)}
    """)

    
    env.start()
    res = env.reset()

     
    ######################################
    ######## Human learning phase ########
    ######################################
    pbar = tqdm(total=total_episodes, desc="Human learning")
    for episode in range(human_learning_episodes):
        env.step()
        pbar.update()
        last_logged_episode = log_new_episodes(
            wb_run, episodes_folder, last_logged_episode, "human_learning", env
        )

    ######### Mutation ########
    # We make object copies, in case they switch back, they will start where they left off
    human_agents_copy = {str(agent.id): copy.deepcopy(agent) for agent in env.human_agents}
    env.mutation(disable_human_learning = not should_humans_adapt, mutation_start_percentile = -1)
    machine_agents_copy = {str(agent.id): copy.deepcopy(agent) for agent in env.machine_agents}

    print(f"""
    Agents in the traffic:
    - Total agents           : {len(env.all_agents)}
    - Human agents           : {len(env.human_agents)}
    - AV agents              : {len(env.machine_agents)}
    """)

    # Replace AV models with baseline models
    machines = env.machine_agents.copy()
    mutated_humans = dict()

    for machine in machines:
        human = human_agents_copy.get(str(machine.id))
        if human is not None:
            mutated_humans[str(machine.id)] = copy.deepcopy(human)
            
    human_learning_params = env.agent_params[kc.HUMAN_PARAMETERS]
    human_learning_params["model"] = baseline_model
    free_flows = env.get_free_flow_times()
    for h_id, human in mutated_humans.items():
        initial_knowledge = free_flows[(human.origin, human.destination)]
        initial_knowledge = [-1 * item for item in initial_knowledge]
        mutated_humans[h_id].model = get_baseline(human_learning_params, initial_knowledge)

    for machine in env.machine_agents:
        machine_id = str(machine.id)
        if machine_id in mutated_humans:
            machine.model = mutated_humans[machine_id].model
       
    ###############################################
    ######## AV learning + Switching phase ########
    ###############################################
    pbar.set_description("AV learning")
    for episode in range(training_eps + dynamic_episodes):
        env.reset()
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()

            if termination or truncation:
                obs = [{kc.AGENT_ID : int(agent), kc.TRAVEL_TIME : -reward}]
                last_action = mutated_humans[agent].last_action
                mutated_humans[agent].learn(last_action, obs)
                action = None
            else:
                action = mutated_humans[agent].act(0)
                mutated_humans[agent].last_action = action

            env.step(action)
           
        ################################## 
        ######## Dynamic switches ########
        ##################################
        # If we are in the dynamic switching phase and the episode is a switch day
        if (episode > training_eps) and (episode % switch_interval == 0):
            shifted_humans, shifted_avs = list(), list()
            
            for human_id in human_agents_copy:
                if human_id not in env.possible_agents:
                    agent_to_copy = next((agent for agent in env.human_agents if str(agent.id) == human_id), None)
                    assert agent_to_copy is not None, f"Human agent {human_id} not found in both possible agents and human agents."
                    human_agents_copy[human_id] = copy.deepcopy(agent_to_copy)
                    
            for machine_id in machine_agents_copy:
                if machine_id in env.possible_agents:
                    agent_to_copy = next((agent for agent in env.machine_agents if str(agent.id) == machine_id), None)
                    assert agent_to_copy is not None, f"AV agent {machine_id} found in possible agents but not in machine agents."
                    machine_agents_copy[machine_id] = copy.deepcopy(agent_to_copy)
            
            known_machines = set(machine_agents_copy.keys())
            
            for human in env.human_agents:
                if random.random() <= switch_prob_humans:
                    env.human_agents.remove(human)
                    env.all_agents.remove(human)
                    
                    if human.id in known_machines:
                        new_av = copy.deepcopy(machine_agents_copy[human.id])
                    else:
                        new_av = MachineAgent(human.id, human.start_time,
                                            human.origin, human.destination,
                                            env.agent_params[kc.MACHINE_PARAMETERS], env.action_space_size)
                    
                    human_id = str(human.id)
                    if human_id not in mutated_humans:
                        baseline_human = copy.deepcopy(human_agents_copy[human_id])
                        initial_knowledge = free_flows[(human.origin, human.destination)]
                        initial_knowledge = [-1 * item for item in initial_knowledge]
                        baseline_human.model = get_baseline(human_learning_params, initial_knowledge)
                        mutated_humans[human_id] = baseline_human
                    
                    new_av.model = mutated_humans[human_id].model
                    env.machine_agents.append(new_av)
                    shifted_humans.append(str(human.id))
                      
            for machine in env.machine_agents:
                if (machine.id not in shifted_humans) and (random.random() <= switch_prob_machines):
                    env.machine_agents.remove(machine)
                    env.all_agents.remove(machine)
                    
                    new_human = copy.deepcopy(human_agents_copy[str(machine.id)])
                    env.human_agents.append(new_human)
                    
                    shifted_avs.append(str(machine.id))
             
            env.all_agents = env.machine_agents + env.human_agents       
            env._initialize_machine_agents()
            # Record switches
            shifted_humans = " ".join(shifted_humans) if shifted_humans else "None"
            shifted_avs = " ".join(shifted_avs) if shifted_avs else "None"
            shifts_df.extend(
                pl.DataFrame({
                "episode": [episode], "shifted_humans": [shifted_humans],
                "shifted_avs": [shifted_avs], "machine_ratio": [len(env.machine_agents) / len(env.all_agents)]
                })
            )
            shifts_df.write_csv(shifts_path)
            ##############################
        pbar.update()
        phase_label = "training" if episode < training_eps else "dynamic"
        last_logged_episode = log_new_episodes(
            wb_run, episodes_folder, last_logged_episode, phase_label, env
        )
    ###############################################
    
    
    ###############################
    ######## Testing phase ########
    ###############################
    pbar.set_description("Testing")
    for episode in range(test_eps):
        env.reset()
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            if termination or truncation:
                action = None
            else:
                action = mutated_humans[agent].act(0)
            env.step(action)
        pbar.update()
        last_logged_episode = log_new_episodes(
            wb_run, episodes_folder, last_logged_episode, "testing", env
        )
    ###############################

    pbar.close()
    env.plot_results()

    env.stop_simulation()

    clear_SUMO_files(os.path.join(records_folder, "SUMO_output"), episodes_folder, remove_additional_files=True)
    finish_runtime_tracking(runtime_tracker)
    ensure_recorder_flush(env)
    last_logged_episode = log_new_episodes(
        wb_run, episodes_folder, last_logged_episode, "final", env
    )
    finish_wandb_run(wb_run, last_logged_episode)
