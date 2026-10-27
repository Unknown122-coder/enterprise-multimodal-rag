"""Deploy to Google Cloud Run from this machine.

Hugging Face now requires a PRO subscription for Docker Spaces, so Cloud Run is
the free-tier host: scale-to-zero, 2 million requests and 360,000 GB-seconds a
month free on any account. The artifact store stays a private Hugging Face
dataset and the vectors stay in Qdrant Cloud; only the container moves.

    python scripts/deploy_cloud_run.py --plan          # check settings, touch nothing
    python scripts/deploy_cloud_run.py                 # do it
    python scripts/deploy_cloud_run.py --skip-store    # redeploy the container only

Needs the gcloud CLI logged in (`gcloud auth login`) and these in .env.deploy:
GCP_PROJECT, GCP_REGION, plus the HF and Qdrant settings from
.env.deploy.example. Steps, each idempotent:

1. verify the HF token, cloud Qdrant and gcloud login
2. create the private dataset repo
3. copy local artifacts and manifests to the dataset repo
4. create the cloud collection and replay the artifacts into it
5. enable the Cloud Run and Artifact Registry APIs, create the image repository
6. build the image locally, tag it, push it to Artifact Registry
7. deploy the service with the environment, capped at one instance
8. wait for /healthz and /readyz on the service URL

Secrets go to the service as environment variables rather than Secret Manager:
simpler, no per-version cost, and visible only to project owners. The service
is public; the API refuses anonymous callers in prod.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
import yaml
from deploy_space import (
    DEPLOY_ENV,
    load_deploy_env,
    say,
    step_artifacts,
    step_dataset,
    step_vectors,
    step_verify,
    tenants_json,
)

ROOT = Path(__file__).resolve().parent.parent
REQUIRED = (
    "HF_TOKEN",
    "HF_DATASET_REPO",
    "QDRANT_URL",
    "QDRANT_API_KEY",
    "GCP_PROJECT",
    "GCP_REGION",
)
IMAGE_REPO = "emrag"


def gcloud_bin() -> str:
    """The gcloud executable, which on Windows is a .cmd shim.

    Also checks the installer's default folders, because a shell opened before
    the install does not see the PATH change.
    """
    for name in ("gcloud", "gcloud.cmd"):
        found = shutil.which(name)
        if found:
            return found
    candidates = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Google" / "Cloud SDK",
        Path(os.environ.get("PROGRAMFILES(X86)", "")) / "Google" / "Cloud SDK",
        Path(os.environ.get("PROGRAMFILES", "")) / "Google" / "Cloud SDK",
    ]
    for base in candidates:
        shim = base / "google-cloud-sdk" / "bin" / "gcloud.cmd"
        if shim.exists():
            return str(shim)
    raise SystemExit(
        "gcloud not found on PATH; install the Google Cloud CLI and reopen the terminal"
    )


def gcloud(*args: str, env: dict[str, str], check: bool = True, quiet: bool = False) -> str:
    """Run one gcloud command and return its stdout."""
    cmd = [gcloud_bin(), *args, "--project", env["GCP_PROJECT"]]
    if quiet:
        cmd.append("--quiet")
    result = subprocess.run(cmd, text=True, capture_output=True)
    if check and result.returncode != 0:
        raise SystemExit(f"   gcloud {' '.join(args[:3])} failed:\n{result.stderr[-1200:]}")
    return result.stdout.strip()


def check_env(env: dict[str, str]) -> list[str]:
    missing = [k for k in REQUIRED if not env.get(k)]
    if not (env.get("GROQ_API_KEY") or env.get("GEMINI_API_KEY")):
        missing.append("GROQ_API_KEY or GEMINI_API_KEY")
    if "localhost" in env.get("QDRANT_URL", "") or "127.0.0.1" in env.get("QDRANT_URL", ""):
        missing.append("QDRANT_URL (must be the cloud cluster, not localhost)")
    return missing


def image_name(env: dict[str, str]) -> str:
    return f"{env['GCP_REGION']}-docker.pkg.dev/{env['GCP_PROJECT']}/{IMAGE_REPO}/emrag:latest"


# --------------------------------------------------------------------------- #


def step_gcloud_check(env: dict[str, str]) -> None:
    say("1b. gcloud")
    account = gcloud("config", "get-value", "account", env=env, check=False)
    if not account:
        raise SystemExit("   gcloud is not logged in; run: gcloud auth login")
    sys.stdout.write(f"   logged in as {account}; project {env['GCP_PROJECT']}\n")


def step_enable(env: dict[str, str]) -> None:
    say("5. Cloud Run and Artifact Registry")
    gcloud("services", "enable", "run.googleapis.com", "artifactregistry.googleapis.com", env=env)
    sys.stdout.write("   APIs enabled\n")
    existing = gcloud(
        "artifacts",
        "repositories",
        "list",
        "--location",
        env["GCP_REGION"],
        "--format",
        "value(name)",
        env=env,
    )
    if IMAGE_REPO not in existing:
        gcloud(
            "artifacts",
            "repositories",
            "create",
            IMAGE_REPO,
            "--repository-format",
            "docker",
            "--location",
            env["GCP_REGION"],
            "--description",
            "emrag images",
            env=env,
        )
        sys.stdout.write(f"   image repository {IMAGE_REPO} created in {env['GCP_REGION']}\n")
    else:
        sys.stdout.write(f"   image repository {IMAGE_REPO} exists\n")
    gcloud("auth", "configure-docker", f"{env['GCP_REGION']}-docker.pkg.dev", env=env, quiet=True)
    sys.stdout.write("   docker authenticated to Artifact Registry\n")


def step_image(env: dict[str, str], *, rebuild: bool) -> None:
    say("6. image", image_name(env))
    local = "emrag:cloudrun"
    cached = subprocess.run(["docker", "image", "inspect", local], capture_output=True)
    if rebuild or cached.returncode != 0:
        sys.stdout.write("   building (first time takes several minutes)...\n")
        result = subprocess.run(["docker", "build", "-t", local, str(ROOT)], text=True)
        if result.returncode != 0:
            raise SystemExit("   docker build failed")
    subprocess.run(["docker", "tag", local, image_name(env)], check=True)
    sys.stdout.write("   pushing 1.8 GB; a few minutes on a normal connection...\n")
    result = subprocess.run(["docker", "push", image_name(env)], text=True)
    if result.returncode != 0:
        raise SystemExit("   docker push failed")
    sys.stdout.write("   pushed\n")


def service_env(env: dict[str, str]) -> dict[str, str]:
    webhook = env.get("HF_WEBHOOK_SECRET") or secrets.token_urlsafe(32)
    if not env.get("HF_WEBHOOK_SECRET"):
        with DEPLOY_ENV.open("a", encoding="utf-8") as fh:
            fh.write(f"\nHF_WEBHOOK_SECRET={webhook}\n")
        sys.stdout.write("   generated HF_WEBHOOK_SECRET and appended it to .env.deploy\n")
    values = {
        "APP_ENV": "prod",
        "ALLOW_DEV_ANON": "false",
        "ARTIFACT_BACKEND": "hf",
        "HF_DATASET_REPO": env["HF_DATASET_REPO"],
        "HF_TOKEN": env["HF_TOKEN"],
        "HF_WEBHOOK_SECRET": webhook,
        "VECTOR_BACKEND": "qdrant",
        "QDRANT_URL": env["QDRANT_URL"],
        "QDRANT_API_KEY": env["QDRANT_API_KEY"],
        "EMBED_BACKEND": "fastembed",
        "RETRIEVAL_GATE_TAU": env.get("RETRIEVAL_GATE_TAU", "0.05"),
        "LOG_LEVEL": env.get("LOG_LEVEL", "INFO"),
        "TENANTS_JSON": tenants_json(env),
    }
    for key in ("GROQ_API_KEY", "GEMINI_API_KEY"):
        if env.get(key):
            values[key] = env[key]
    return values


def step_deploy(env: dict[str, str]) -> str:
    service = env.get("CLOUD_RUN_SERVICE", "emrag")
    say("7. deploy service", f"{service} in {env['GCP_REGION']}, 2 GiB, max 1 instance")
    values = service_env(env)
    with tempfile.TemporaryDirectory() as tmp:
        env_file = Path(tmp) / "env.yaml"
        env_file.write_text(yaml.safe_dump(values, default_style='"'), encoding="utf-8")
        gcloud(
            "run",
            "deploy",
            service,
            "--image",
            image_name(env),
            "--region",
            env["GCP_REGION"],
            "--platform",
            "managed",
            "--port",
            "7860",
            "--memory",
            "2Gi",
            "--cpu",
            "1",
            "--cpu-boost",
            "--min-instances",
            "0",
            "--max-instances",
            env.get("CLOUD_RUN_MAX_INSTANCES", "1"),
            "--concurrency",
            "4",
            "--timeout",
            "300",
            "--allow-unauthenticated",
            "--env-vars-file",
            str(env_file),
            env=env,
            quiet=True,
        )
    url = gcloud(
        "run",
        "services",
        "describe",
        service,
        "--region",
        env["GCP_REGION"],
        "--format",
        "value(status.url)",
        env=env,
    )
    sys.stdout.write(f"   deployed: {url}\n")
    sys.stdout.write(f"   env keys set: {', '.join(sorted(values))}\n")
    return url


def step_wait(url: str, timeout_s: int) -> None:
    say("8. wait for the service", f"{url}/healthz")
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            if httpx.get(url + "/healthz", timeout=30).status_code == 200:
                break
        except httpx.HTTPError:
            pass
        sys.stdout.write(".")
        sys.stdout.flush()
        time.sleep(10)
    else:
        raise SystemExit("   service did not become healthy; check Cloud Run logs in the console")
    sys.stdout.write("\n   /healthz 200\n")
    for _ in range(30):
        try:
            r = httpx.get(url + "/readyz", timeout=30)
            if r.status_code == 200 and r.json().get("ready"):
                body = r.json()
                sys.stdout.write(f"   /readyz ready; detail={json.dumps(body.get('detail'))}\n")
                break
            sys.stdout.write(f"   /readyz checks={json.dumps(r.json().get('checks'))}\n")
        except httpx.HTTPError:
            pass
        time.sleep(10)
    else:
        sys.stdout.write("   /readyz not ready yet; see the checks above\n")
    sys.stdout.write(
        "\nNext:\n"
        f"  curl -sS {url}/v1/meta\n"
        f"  register the dataset webhook: {url}/v1/internal/hf-webhook with X-Webhook-Secret\n"
        "  set a budget alert at 1 USD in the Cloud console\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--plan", action="store_true", help="print the plan and exit")
    parser.add_argument("--skip-store", action="store_true", help="skip artifact and vector sync")
    parser.add_argument("--rebuild", action="store_true", help="rebuild the image even if cached")
    parser.add_argument("--wait", type=int, default=600, help="seconds to wait for /healthz")
    args = parser.parse_args()

    env = load_deploy_env()
    missing = check_env(env)
    sys.stdout.write(
        f"project={env.get('GCP_PROJECT', '?')}  region={env.get('GCP_REGION', '?')}  "
        f"dataset={env.get('HF_DATASET_REPO', '?')}\n"
    )
    if missing:
        sys.stdout.write("missing: " + ", ".join(missing) + "\n")
        return 2
    if args.plan:
        sys.stdout.write(
            "plan: verify -> dataset -> artifacts -> vectors -> enable APIs -> image "
            "-> deploy -> wait\n"
        )
        return 0

    step_verify(env)
    step_gcloud_check(env)
    step_dataset(env)
    if not args.skip_store:
        step_artifacts(env)
        step_vectors(env)
    step_enable(env)
    step_image(env, rebuild=args.rebuild)
    url = step_deploy(env)
    step_wait(url, args.wait)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    raise SystemExit(main())
