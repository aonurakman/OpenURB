import os
import pytest
import shutil
import subprocess

from pathlib import Path

# Executes every SUMO-aware script in the repository with a minimal config.

SCRIPTS_DIR = Path("scripts")
python_scripts = list(SCRIPTS_DIR.rglob("*.py"))
excluded_scripts = ["utils.py", "base_script.py", "baselines.py", "open_iql.py", "cond_open_iql.py", "parallel_open_iql.py"]

print(f"[DEBUG] Looking for Python scripts in {SCRIPTS_DIR.resolve()}")
print(f"[DEBUG] Found {len(python_scripts)} Python scripts (excluding {len(excluded_scripts)} scripts).")
python_scripts = [script for script in python_scripts if script.name not in excluded_scripts]

@pytest.fixture(scope="session", autouse=True)
def check_sumo_installed():
    # Treat missing SUMO as a skip so the suite can still run elsewhere.
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
def test_python_script_execution(script_path):
    """Smoke test each script to ensure CLI arguments are accepted."""
    try:
        script_filename = script_path.name
        print(script_filename)
        result = subprocess.run(
            ["python", script_filename,
             "--id", f"test_{script_filename}", 
             "--alg-conf", "test", 
             "--env-conf", "test", 
             "--task-conf", "test", 
             "--net", "saint_arnoult"],
            capture_output=True, text=True, check=True, cwd=script_path.parent
        )
        print(f"[DEBUG] Successfully executed {script_path}")
    except subprocess.CalledProcessError as e:
        pytest.fail(f"[FAIL] Script {script_path} failed to execute: {e.stderr}")
