#!/usr/bin/env python3

from redis import Redis
from rq import Queue
import logging
import sys
import os
from pathlib import Path


# Import the run_pipeline_in_apptainer function from runqueue.py:
from runqueue import run_pipeline_in_apptainer

# This script is used to enqueue the pipeline for a list of subjects and sessions.
# It reads from the file passed as the first argument, where each line is a subject and session separated by a space.
# The session is then enqueued to be processed by the pipeline.

logging.basicConfig(level=logging.INFO)

# Get the path to the file containing the list of subjects and sessions:
if len(sys.argv) != 2:
    raise ValueError(
        "Please provide the path to the file containing the list of subjects and sessions as the first argument"
    )

p = Path(sys.argv[1]).expanduser().resolve()
if not p.exists():
    raise FileNotFoundError(f"Could not find the file at {p}")

# Set up the rq queue:
q = Queue(connection=Redis(), default_timeout=(3600 * 6))


sif_path = Path(os.environ.setdefault("DHCP_PIPELINE_SIF_PATH", "szika-mrtrixfs-extras.sif")).resolve()
if not sif_path.exists():
    raise FileNotFoundError(f"Could not find the Apptainer image file at {sif_path}")

logging.info(f"Using Apptainer image file {sif_path}")

# Read the file and enqueue each subject and session
# Use dict to remove duplicates while preserving order:
for subject, session in tuple({tuple(x.split()): None for x in p.read_text().strip().split("\n")}.keys()):
    # Call the run_pipeline_in_apptainer function with the subject and session as arguments:
    q.enqueue(run_pipeline_in_apptainer, sif_path, subject, session)
    logging.info(f"Enqueued {subject} {session}")
