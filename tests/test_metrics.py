import sys
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import pytest

from analysis import metrics

# Validates metrics helpers against the canned SUMO sample results.

SAMPLE_RESULTS_DIR = PROJECT_ROOT / "results" / "sample_results"
EPISODES_DIR = SAMPLE_RESULTS_DIR / "episodes"
SUMO_OUTPUT_DIR = SAMPLE_RESULTS_DIR / "SUMO_output"


def test_get_episodes_matches_sample_results():
    """Ensure get_episodes discovers the full range of exported CSVs."""
    episodes = metrics.get_episodes(str(EPISODES_DIR))
    assert episodes[0] == 1
    assert episodes[-1] == 30
    assert len(episodes) == 30


def test_flatten_by_id_on_sample_episode():
    """Flattening should yield a single row with agent-prefixed columns."""
    episode_df = pd.read_csv(EPISODES_DIR / "ep1.csv")
    flattened = metrics.flatten_by_id(episode_df)

    assert flattened.shape[0] == 1
    assert flattened.iloc[0]["agent_0_action"] == 0
    assert flattened.iloc[0]["agent_3_reward"] == pytest.approx(-1.8)


def test_add_benchmark_columns_uses_sample_data():
    """Benchmark feature engineering should populate action and time deltas."""
    episode_flat = metrics.flatten_by_id(pd.read_csv(EPISODES_DIR / "ep1.csv"))
    detailed_flat = metrics.load_detailed_SUMO(str(SUMO_OUTPUT_DIR / "detailed_sumo_stats_1.xml"))
    general_df = metrics.load_general_SUMO(str(SUMO_OUTPUT_DIR / "sumo_stats_1.xml"))

    combined = pd.concat([episode_flat, detailed_flat, general_df], axis=1)
    assert "agent_0_duration" in combined.columns

    params = {
        "avg_times_pre": {
            0: combined.get("agent_0_duration", pd.Series([0.0])).iloc[0],
        }
    }
    augmented = metrics.add_benchmark_columns(combined, params)

    assert "agent_0_action_change" in augmented.columns
    assert "agent_0_time_lost" in augmented.columns
    assert augmented.iloc[0]["agent_0_action_change"] == 1
    assert augmented.iloc[0]["agent_0_time_lost"] == pytest.approx(0.0)


def test_get_type_ids_reads_from_sample_sumo():
    """SUMO tripinfo data should report agent IDs for the given vehicle type."""
    detailed_flat = metrics.load_detailed_SUMO(str(SUMO_OUTPUT_DIR / "detailed_sumo_stats_1.xml"))
    ids = metrics.get_type_ids(detailed_flat, "Human")
    assert ids, "Expected at least one Human vType in sample detailed SUMO output"


def test_load_general_and_detailed_sumo_from_sample_results():
    """Loading helpers must return non-empty frames for provided XML files."""
    detailed_df = metrics.load_detailed_SUMO(str(SUMO_OUTPUT_DIR / "detailed_sumo_stats_1.xml"))
    general_df = metrics.load_general_SUMO(str(SUMO_OUTPUT_DIR / "sumo_stats_1.xml"))

    assert not detailed_df.empty
    assert not general_df.empty


def test_process_experiment_forwards_effective_human_learning_episodes(tmp_path, monkeypatch):
    """Pretrained-human runs must forward effective human-learning episodes to metrics extraction."""
    exp_dir = tmp_path / "exp"
    metrics_dir = exp_dir / "metrics"
    metrics_dir.mkdir(parents=True)

    exp_config = {
        "human_learning_episodes": 80,
        "effective_human_learning_episodes": 0,
        "training_eps": 240,
        "dynamic_episodes": 240,
        "test_eps": 25,
    }
    (exp_dir / "exp_config.json").write_text(json.dumps(exp_config), encoding="utf-8")
    (metrics_dir / "combined_data.csv").write_text("episode\n1\n", encoding="utf-8")

    captured: dict = {}

    def fake_extract_metrics(path, config, verbose=False, shifts_path=None):
        captured["path"] = path
        captured["config"] = dict(config)
        return pd.DataFrame([{"t_CAV": 1.0}]), pd.DataFrame()

    monkeypatch.setattr(metrics, "extract_metrics", fake_extract_metrics)

    status = metrics.process_experiment(
        exp_id="exp",
        data_path=str(exp_dir),
        skip_collecting=True,
        no_skip=True,
        verbose=False,
    )

    assert status == "success"
    assert captured["config"]["effective_human_learning_episodes"] == 0
