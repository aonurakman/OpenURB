import xml.etree.ElementTree as ET
import glob
import os
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Any, Dict

try:
    import resource
except ImportError:  # pragma: no cover - Windows compatibility
    resource = None
try:
    import wandb
except ImportError:
    wandb = None


@dataclass
class RuntimeTracker:
    records_folder: str
    exp_id: str
    script_path: str
    alg_config: Optional[str]
    task_config: Optional[str]
    env_config: Optional[str]
    start_epoch: float
    start_dt: datetime
    start_self_usage: Optional[Any]
    start_child_usage: Optional[Any]


def start_runtime_tracking(records_folder: str, exp_id: str, script_path: str,
                           alg_config: Optional[str] = None, task_config: Optional[str] = None,
                           env_config: Optional[str] = None) -> RuntimeTracker:
    start_epoch = time.time()
    start_dt = datetime.now().astimezone()
    start_self_usage = resource.getrusage(resource.RUSAGE_SELF) if resource is not None else None
    start_child_usage = resource.getrusage(resource.RUSAGE_CHILDREN) if resource is not None else None
    return RuntimeTracker(records_folder, exp_id, script_path, alg_config, task_config,
                          env_config, start_epoch, start_dt, start_self_usage, start_child_usage)


def _format_duration(seconds: float) -> str:
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def finish_runtime_tracking(runtime_tracker: RuntimeTracker) -> None:
    if runtime_tracker is None:
        return

    runtime_path = os.path.join(runtime_tracker.records_folder, "runtime.json")
    end_epoch = time.time()
    end_dt = datetime.now().astimezone()
    total_runtime_sec = end_epoch - runtime_tracker.start_epoch

    payload = {
        "experiment": {
            "id": runtime_tracker.exp_id,
            "script": os.path.basename(runtime_tracker.script_path),
            "alg_config": runtime_tracker.alg_config,
            "task_config": runtime_tracker.task_config,
            "env_config": runtime_tracker.env_config,
        },
        "timing": {
            "start_local": runtime_tracker.start_dt.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "end_local": end_dt.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "duration": _format_duration(total_runtime_sec),
        },
    }

    if resource is not None and runtime_tracker.start_self_usage and runtime_tracker.start_child_usage:
        end_self = resource.getrusage(resource.RUSAGE_SELF)
        end_child = resource.getrusage(resource.RUSAGE_CHILDREN)
        resources = payload.setdefault("resources", {})
        resources.update(
            {
                "cpu_user_sec": round(end_self.ru_utime - runtime_tracker.start_self_usage.ru_utime, 3),
                "cpu_system_sec": round(end_self.ru_stime - runtime_tracker.start_self_usage.ru_stime, 3),
                "voluntary_context_switches": end_self.ru_nvcsw - runtime_tracker.start_self_usage.ru_nvcsw,
                "involuntary_context_switches": end_self.ru_nivcsw - runtime_tracker.start_self_usage.ru_nivcsw,
            }
        )

        def _rss_bytes(usage):
            rss = usage.ru_maxrss
            if sys.platform != "darwin":
                rss *= 1024
            return rss

        max_rss_bytes = max(_rss_bytes(end_self), _rss_bytes(end_child))
        resources["peak_memory_mb"] = round(max_rss_bytes / (1024 * 1024), 3)

    folder_bytes = 0
    for root, _, files in os.walk(runtime_tracker.records_folder):
        for filename in files:
            filepath = os.path.join(root, filename)
            try:
                folder_bytes += os.path.getsize(filepath)
            except OSError:
                pass
    payload.setdefault("resources", {})["records_folder_size_mb"] = round(folder_bytes / (1024 * 1024), 3)

    with open(runtime_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=4)


def load_wandb_secrets(config_path: str) -> Dict[str, str]:
    """Load W&B credentials/project defaults from json if it exists."""
    if not os.path.exists(config_path):
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def init_wandb_run(wandb_config_path: str, run_name: str, config: Dict[str, Any],
                   disabled: bool = False):
    """Initialize a Weights & Biases run and return the run handle (or None)."""
    if disabled:
        return None
    if wandb is None:
        raise ImportError("wandb is not installed. Please install wandb or use --no-wandb.")
    wb_secrets = load_wandb_secrets(wandb_config_path)
    api_key = wb_secrets.get("api_key")
    if api_key:
        wandb.login(key=api_key, relogin=True)
    wb_kwargs = {
        "project": wb_secrets.get("project", "openurb"),
        "entity": wb_secrets.get("entity"),
        "name": run_name,
        "config": config,
    }
    wb_kwargs = {k: v for k, v in wb_kwargs.items() if v is not None}
    wb_run = wandb.init(**wb_kwargs)
    wb_run.log({"status": "started"}, step=0)
    return wb_run


def ensure_recorder_flush(env) -> None:
    """Wait for any pending async Recorder writes so episode files are ready."""
    pending = getattr(env, "pending_futures", None)
    if not pending:
        return
    for future in list(pending):
        future.result()
    env.pending_futures = []


def log_new_episodes(wb_run, episodes_folder: str, last_logged: int,
                     phase: str, env) -> int:
    """Log per-episode mean rewards/travel_times grouped by agent kind."""
    if wb_run is None:
        return last_logged

    ensure_recorder_flush(env)

    ep_files = glob.glob(os.path.join(episodes_folder, "ep*.csv"))
    if not ep_files:
        return last_logged

    try:
        import pandas as pd
    except ImportError:
        return last_logged

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
            kind_key = str(kind).lower()
            metrics[f"{phase}/reward/{kind_key}"] = group["reward"].mean()
            metrics[f"{phase}/travel_time/{kind_key}"] = group["travel_time"].mean()
        wb_run.log(metrics, step=ep_num)
        last_logged = ep_num

    return last_logged


def finish_wandb_run(wb_run, last_logged: int) -> None:
    if wb_run is None:
        return
    wb_run.log({"status": "finished"}, step=last_logged + 1)
    wb_run.finish()

def get_episodes(ep_path: str) -> list[int]:
    """Get the episodes data

    Returns:
        sorted_episodes (list[int]): the sorted episodes data
    Raises:
        FileNotFoundError: If the episodes folder does not exist
    """

    eps = list()
    if os.path.exists(ep_path):
        for file in os.listdir(ep_path):
            episode = int(file.split("ep")[1].split(".csv")[0])
            eps.append(episode)
    else:
        raise FileNotFoundError(f"Episodes folder does not exist!")


    return sorted(eps)


def clear_SUMO_files(sumo_path, ep_path, remove_additional_files=False):
    '''
        Clear SUMO files that are empty or not in the episodes folder.
        Works only for the consecutive files with the same name.
        The files are named as <file_name>_<episode>.xml

        This is a destructive function, it will remove files from the directory!
    '''
    file_id = 1
    episode = 1

    file_name = "detailed_sumo_stats"
    
    while True:
        # check if file exists
        file_path = os.path.join(sumo_path, f"{file_name}_{episode}.xml")
        if os.path.exists(file_path):
            # read xml file and check if <tripinfos> is empty (no <tripinfo> elements)
            try:
                tree = ET.parse(file_path)
            except ET.ParseError:
                print(f"Error parsing XML file: {file_path}")
                break
            root = tree.getroot()
            if len(root.findall("tripinfo")) == 0:
                # remove the file
                os.remove(file_path)
                # print(f"Removed empty file: {file_path}")
            else:
                # rename to the next file_id
                new_file_path = os.path.join(sumo_path, f"{file_name}_{file_id}.xml")
                os.rename(file_path, new_file_path)
                # print(f"Renamed file {file_path} to {new_file_path}")
                file_id += 1
        else:
            break
        episode += 1

    file_id = 1
    episode = 1

    file_name = "sumo_stats"

    while True:
        # check if file exists
        file_path = os.path.join(sumo_path, f"{file_name}_{episode}.xml")
        if os.path.exists(file_path):
            # read xml file and check if <vehicle loaded=0>
            try:
                tree = ET.parse(file_path)
            except ET.ParseError:
                print(f"Error parsing XML file: {file_path}")
                break
            root = tree.getroot()
            vehicle = root.find("vehicles")
            if vehicle is not None and vehicle.attrib.get("loaded") == "0":
                # remove the file
                os.remove(file_path)
            else:
                # rename to the next file_id
                new_file_path = os.path.join(sumo_path, f"{file_name}_{file_id}.xml")
                os.rename(file_path, new_file_path)
                file_id += 1
        else:
            break
        episode += 1
    if remove_additional_files:
        episodes = get_episodes(ep_path)
        # remove SUMO files that are not in the episodes
        for file in os.listdir(sumo_path):
            if file.endswith(".xml"):
                episode = int(file.split("_")[-1].split(".")[0])
                if episode not in episodes:
                    os.remove(os.path.join(sumo_path, file))
                    
                    
def print_agent_counts(env):
    print(f"""
    ----------------------------------------------------
                    Agents in traffic
    ----------------------------------------------------
    Total agents           | {len(env.all_agents)}
    Human agents           | {len(env.human_agents)}
    AV agents              | {len(env.machine_agents)}
    ----------------------------------------------------
    """)
