#!/usr/bin/env python3

import os
import sys
import time
from pathlib import Path
import sh

# Launch the pipeline in Apptainer:
def run_pipeline_in_apptainer(
    sif: str, subject: str, session: str, *args, script_path: str = "endpoints_pipeline.py", **kwargs
):
    subject = str(subject).strip().removeprefix("sub-")
    session = str(session).strip().removeprefix("ses-")
    kwargs.setdefault("_out", "/dev/stdout")
    kwargs.setdefault("_err", "/dev/stderr")
    kwargs.setdefault("_tty_out", False)
    start_time = time.time()
    print("Running pipeline in apptainer with", sif, subject, session, args, script_path, kwargs, sys.stderr)
    cwd_abs = Path.cwd().expanduser().resolve()
    script_abs = cwd_abs / script_path
    if not script_abs.exists():
        raise FileNotFoundError(f"Script not found: {script_abs}")
    cmd = sh.time
    if os.environ.get("DHCP_PIPELINE_CPUSET", ""):
        cmd = cmd.taskset.bake("--cpu-list", os.environ["DHCP_PIPELINE_CPU_LIST"])
    cmd = cmd.apptainer.exec.bake("--env-file", ".env", sif, "python3", script_abs)
    result = cmd(subject, session, *args, **kwargs)  # type: ignore
    print("Pipeline run in apptainer completed", sys.stderr)
    return dict(subject=subject, session=session, elapsed=time.time() - start_time)

