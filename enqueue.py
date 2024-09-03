#!/usr/bin/env python3

from redis import Redis
from rq import Queue
import fsspec
from loguru import logger

import sys
import os
from pathlib import Path

# Import the run_pipeline_in_apptainer function from runqueue.py:
from runqueue import run_pipeline_in_apptainer

# This script is used to enqueue the pipeline for a list of subjects and sessions.
# It reads from the file passed as the first argument, where each line is a subject and session separated by a space.
# The session is then enqueued to be processed by the pipeline.

# Get the path to the file containing the list of subjects and sessions:
p = os.path.abspath(sys.argv[1])

# Set up the rq queue:
q = Queue(connection=Redis(), default_timeout=(3600 * 6))

logger.info("Starting enqueuing")

os.environ.setdefault("DHCP_PIPELINE_SIF_PATH", "endpoints_pipeline.sif")

sif_path = Path(os.environ["DHCP_PIPELINE_SIF_PATH"]).resolve()
if not sif_path.exists():
    raise FileNotFoundError(f"Could not find the Apptainer image file at {sif_path}")

logger.info(f"Using Apptainer image file {sif_path}")

# Read the file and enqueue each subject and session:
for subject, session in [tuple(x.split()) for x in open(p).read().splitlines()]:
    # Call the run_pipeline_in_apptainer function with the subject and session as arguments:
    q.enqueue(run_pipeline_in_apptainer, sif_path, subject, session)
    logger.info(f"Enqueued {subject} {session}")

