import os
import pytest
import shutil
import subprocess

from pathlib import Path

# Covers the OPEN-IQL entry points that interact with SUMO.

SCRIPTS_DIR = Path("scripts")
python_script = [SCRIPTS_DIR / "open_iql.py", SCRIPTS_DIR / "cond_open_iql.py"]

@pytest.fixture(scope="session", autouse=True)
def check_sumo_installed():
    # Skip integration tests when the simulator is missing.
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


@pytest.mark.parametrize("script_path", python_script)
def test_python_script_execution(script_path):
    """Validate that condensed and standard OPEN scripts accept test configs."""
    try:
        script_filename = script_path.name
        result = subprocess.run(
            ["python", script_filename,
             "--id", f"test_{script_filename}",
             "--alg-conf", "test",
             "--env-conf", "test",
             "--task-conf", "dynamic_test",
             "--net", "saint_arnoult"],
            capture_output=True, text=True, check=True, cwd=script_path.parent
        )
        print(f"[DEBUG] Successfully executed {script_path}")
    except subprocess.CalledProcessError as e:
        pytest.fail(f"[FAIL] {script_path} failed: {e.stderr}")
