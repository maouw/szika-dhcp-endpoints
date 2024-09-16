#!/usr/bin/env python3

import os
import sys
import time
from pathlib import Path
import logging
import sh


# Launch the pipeline in Apptainer:
def run_pipeline_in_apptainer(
    sif: str, subject: str, session: str, *args, script_path: str = "endpoints_pipeline.py", **kwargs
):
    logging.basicConfig(level=logging.INFO)
    subject = str(subject).strip().removeprefix("sub-")
    session = str(session).strip().removeprefix("ses-")
    kwargs.setdefault("_out", "/dev/stdout")
    kwargs.setdefault("_err", "/dev/stderr")
    kwargs.setdefault("_tty_out", False)
    start_time = time.time()

    cwd_abs = Path.cwd().expanduser().resolve()
    script_abs = cwd_abs / script_path
    if not script_abs.exists():
        raise FileNotFoundError(f"Script not found: {script_abs}")
    cmd = sh.time
    logging.info(
        f"Running pipeline in Apptainer with {sif=}, {subject=}, {session=}, {args=}, {script_path=}, {kwargs=}"
    )
    logging.info(
        f"Running pipeline in Apptainer with {sif=}, {subject=}, {session=}, {args=}, {script_path=}, {kwargs=}"
    )

    # Set the CPU affinity for the container if the DHCP_PIPELINE_CPUSET environment variable is set (should be a comma-separated list of CPU cores):
    if cpuset := os.environ.get("DHCP_PIPELINE_CPUSET", ""):
        cmd = cmd.taskset.bake("--cpu-list", cpuset)  # type: ignore
        logging.info(f"Set CPU affinity to {cpuset}")

    # Set up the command to run the script in the Apptainer container:
    cmd = cmd.apptainer.exec.bake("--env-file", ".env", sif, "python3", script_abs)  # type: ignore

    # Run the command with the subject and session as arguments:
    result = cmd(subject, session, *args, **kwargs)  # type: ignore
    logging.info("Pipeline run in apptainer completed")
    return dict(subject=subject, session=session, elapsed=time.time() - start_time)
