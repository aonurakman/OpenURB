import os
import pytest
import shutil
import subprocess

from pathlib import Path

# Exercises the OPEN baseline runners for every registered baseline model.

SCRIPTS_DIR = Path("scripts")
python_scripts = [SCRIPTS_DIR / "open_baselines.py", SCRIPTS_DIR / "cond_open_baselines.py"]

BASELINES_DIR = Path("baseline_models")
baseline_names = list(BASELINES_DIR.rglob("*.py"))
baseline_names = [name for name in baseline_names if name.name not in ["__init__.py", "base.py", "registry.py"]]

@pytest.fixture(scope="session", autouse=True)
def check_sumo_installed():
    # Bail out early when SUMO is not available in the environment.
    sumo_executable = shutil.which("sumo")
    if sumo_executable is None:
        pytest.skip("[SUMO SKIP] SUMO is not installed or not in PATH.", allow_module_level=True)
    else:
        try:
            result = subprocess.run(
                ["sumo", "--version"], capture_output=True, text=True, check=True
            )
            print(f"[DEBUG] SUMO version: {result.stdout.strip()}")
        except subprocess.CalledProcessError as e:
            pytest.skip(f"[SUMO SKIP] Failed to get SUMO version: {e.stderr}", allow_module_level=True)


@pytest.mark.parametrize("script_path", python_scripts)
@pytest.mark.parametrize("baseline", baseline_names)
def test_python_script_execution(script_path, baseline):
    """Run the open baselines launchers for each baseline script entry."""
    try:
        script_filename = script_path.name
        baseline_name = baseline.name.split(".")[0]
        result = subprocess.run(
            ["python", script_filename,
             "--id", f"test_{script_filename}_{baseline_name}",
             "--alg-conf", "test",
             "--env-conf", "test",
             "--task-conf", "test",
             "--net", "saint_arnoult",
             "--model", baseline_name,
             "--no-wandb"],
            capture_output=True, text=True, check=True, cwd=script_path.parent
        )
        print(f"[DEBUG] Successfully executed baseline {baseline_name} with {script_path}")
    except subprocess.CalledProcessError as e:
        pytest.fail(f"[FAIL] Baseline {baseline_name} failed for {script_path}: {e.stderr}")
