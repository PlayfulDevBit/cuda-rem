"""
deploy.py — Prefect Deployment Configuration
==============================================
This file registers the pipeline with Prefect so it can be:
    - Triggered manually from the Prefect UI or CLI
    - Scheduled to run automatically
    - Run on a remote worker with the correct Docker image

YOU DO NOT NEED TO EDIT THIS FILE unless you are changing:
    - The schedule
    - The Docker image
    - The Prefect work pool
    - Environment variable handling

USAGE:
    python deploy.py          # register / update the deployment on Prefect
    prefect deployment run qaoa-iqm-resonance/iqm-resonance-deploy  # trigger it
"""

from prefect.deployments import Deployment
from prefect.infrastructure import DockerContainer
from prefect.server.schemas.schedules import CronSchedule

from qaoa_iqm_pipeline import qaoa_iqm_flow

# ─────────────────────────────────────────────
# INFRASTRUCTURE
# Docker image must have installed:
#   cuda-quantum, prefect, scipy, networkx, numpy
# ─────────────────────────────────────────────

docker_infra = DockerContainer(
    image="your-registry/qaoa-iqm:latest",   # ← replace with your image
    image_pull_policy="ALWAYS",
    env={
        "IQM_TOKEN":      "{{ $IQM_TOKEN }}",        # injected from Prefect secrets
        "IQM_SERVER_URL": "{{ $IQM_SERVER_URL }}",
    },
    networks=["host"],   # required for IQM Resonance network access
)

# ─────────────────────────────────────────────
# DEPLOYMENT
# ─────────────────────────────────────────────

deployment = Deployment.build_from_flow(
    flow=qaoa_iqm_flow,
    name="iqm-resonance-deploy",
    version="1.0",
    work_queue_name="quantum",          # match your Prefect work pool name
    infrastructure=docker_infra,

    # Default parameters — override at trigger time from UI or CLI
    parameters={
        "config": {
            "n_qubits": 4,
            "p_layers": 2,
            "shots": 4096,
            "cobyla_max_iter": 150,
            "cobyla_rhobeg": 0.5,
            "rem_shots": 1024,
            "seed": 42,
        }
    },

    # Optional: run every day at 08:00 UTC
    # schedule=CronSchedule(cron="0 8 * * *", timezone="UTC"),
    schedule=None,   # manual trigger by default
)

if __name__ == "__main__":
    deployment_id = deployment.apply()
    print(f"✅  Deployment registered: {deployment_id}")
    print("    Trigger with:")
    print("    prefect deployment run qaoa-iqm-resonance/iqm-resonance-deploy")
