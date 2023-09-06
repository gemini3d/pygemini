from __future__ import annotations
import os
import subprocess
import binascii
from pathlib import Path
import shutil
import importlib.resources


def hpc_submit_job(batcher: str, job_file: Path):
    """
    submits batch job
    """

    if batcher == "qsub":
        subprocess.run(["qsub", str(job_file)])
    else:
        raise LookupError(
            f"batcher {batcher} not yet listed. Please raise a Gemini Github Issue."
        )


def hpc_batch_create(batcher: str, out_dir: Path, cmd: list[str]) -> Path:
    """
    creates HPC batch scripts for known systems

    assumes that user-specific parameters like account number are already set
    as environment variables
    or static configuration files not handled by this scripts.

    This function assumes a script template exists, and it merely appends lines
    to the end of that template.

    TODO:

    1. determine estimated wallclock time to request on HPC
    2. determine requested HPC RAM per node (limit is main node)
    3. format number of nodes request
    """

    Nchar = 6  # arbitrary number of characters

    if batcher == "qsub":
        template = (
            importlib.resources.files("gemini3d.templates")
            .joinpath("qsub_template.job")
            .read_text()
        )
        job_file = out_dir / f"{binascii.b2a_hex(os.urandom(Nchar)).decode('ascii')}.job"
        print("writing job file", job_file)
        text = template + "\n" + " ".join(cmd)
        job_file.write_text(text)
    else:
        raise LookupError(
            f"batcher {batcher} not yet listed. Please raise a Gemini Github Issue."
        )

    return job_file


def hpc_batch_detect() -> str | None:
    """
    Assuming a known job batching system, we will create a template for the user
    to verify and then the user will run.
    """

    batcher = None

    if shutil.which("qsub"):
        batcher = "qsub"

    return batcher
