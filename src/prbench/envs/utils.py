"""Utility functions for process management in PRBench environments.

This module provides helpers for creating and cleaning up PID files to
ensure that only one instance of a process is running at a time.
"""

import atexit
import os
from pathlib import Path


def create_pid_file(name):
    """Create a PID file in /tmp to prevent multiple instances of the same
    process.

    If a PID file already exists and the process is still running, raises an Exception.
    If the PID file is stale, it is removed. The PID file is automatically cleaned up
    on process exit.

    Args:
        name (str): Name of the process (used for the PID file name).
    """
    # Check if PID file already exists
    pid_file_path = Path(f"/tmp/{name}.pid")
    if pid_file_path.exists():
        # Get PID of other process from lock file
        with open(pid_file_path, "r", encoding="utf-8") as f:
            pid = int(f.read().strip())

        # Check if PID matches current process
        if pid != os.getpid():
            # PID does not match current process, check if other process is still running
            try:
                os.kill(pid, 0)
            except OSError:
                print(f"Removing stale PID file (PID {pid})")
                pid_file_path.unlink()
            else:
                raise Exception(
                    f"Another instance of the {name} is already running (PID {pid})"
                )

    # Write PID of current process to the file
    pid_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pid_file_path, "w", encoding="utf-8") as f:
        f.write(f"{os.getpid()}\n")

    # Register cleanup function to remove PID file upon exit
    atexit.register(remove_pid_file, pid_file_path)


def remove_pid_file(pid_file_path):
    """Remove the PID file if it corresponds to the current process.

    Args:
        pid_file_path (Path): Path to the PID file to remove.
    """
    # Remove PID file if it corresponds to the current process
    if pid_file_path.exists():
        with open(pid_file_path, "r", encoding="utf-8") as f:
            pid = int(f.read().strip())
        if pid == os.getpid():
            pid_file_path.unlink()
