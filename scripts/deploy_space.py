"""Deploy to a Hugging Face Space from this machine, without a GitHub remote.

The CI workflow in .github/workflows/deploy-hf-space.yml does the same push on a
tag; this script is the same sequence driven locally, for a repository that has
no GitHub remote yet or an operator who wants to watch each step.

    python scripts/deploy_space.py --plan            # show what would happen, touch nothing
    python scripts/deploy_space.py                   # do it
    python scripts/deploy_space.py --skip-store      # redeploy code only

Reads deployment-only settings from ``.env.deploy`` (gitignored; see
``.env.deploy.example``) on top of the normal ``.env``. Steps, each idempotent:

1. verify the HF token and the cloud Qdrant cluster
2. create the private dataset repo and the public Docker Space
3. copy local artifacts and manifests to the dataset repo
4. create the collection in cloud Qdrant and replay the artifacts into it
5. set the Space's secrets and variables
6. push the working tree to the Space with the Space README front matter
7. wait for /healthz, then /readyz, and print what to check

Secrets are read from the environment and never printed. The Space is public
because a private Space requires Hugging Face credentials on every request,
which would collide with the API's own key header; the API itself refuses
anonymous callers in prod.
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

ROOT = Path(__file__).resolve().parent.parent
DEPLOY_ENV = ROOT / ".env.deploy"
STORE_ROOT = ROOT / "var" / "store"
SPACE_PREFIXES = ("artifacts", "manifest")

REQUIRED = ("HF_TOKEN", "HF_SPACE", "HF_DATASET_REPO", "QDRANT_URL", "QDRANT_API_KEY")


def _read_env_file(path: Path, into: dict[str, str]) -> None:
    """Parse KEY=value lines (comments and blanks ignored) into ``into``."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        value = value.split("  #")[0].strip().strip('"').strip("'")
        if value:
            into[key.strip()] = value


def load_deploy_env() -> dict[str, str]:
    """Process environment, then .env, then .env.deploy; later sources win.

    The provider keys live in .env (the app loads it through pydantic); the
    deployment-only values live in .env.deploy. Both are needed here.
    """
    env = dict(os.environ)
    _read_env_file(ROOT / ".env", env)
    _read_env_file(DEPLOY_ENV, env)
    return env


def say(step: str, detail: str = "") -> None:
    sys.stdout.write(f"\n== {step}\n" + (f"   {detail}\n" if detail else ""))
    sys.stdout.flush()


def check_env(env: dict[str, str]) -> list[str]:
    """Return the names of required settings that are missing."""
    missing = [k for k in REQUIRED if not env.get(k)]
    if not (env.get("GROQ_API_KEY") or env.get("GEMINI_API_KEY")):
        missing.append("GROQ_API_KEY or GEMINI_API_KEY")
    if "localhost" in env.get("QDRANT_URL", "") or "127.0.0.1" in env.get("QDRANT_URL", ""):
        missing.append("QDRANT_URL (must be the cloud cluster, not localhost)")
    return missing


def tenants_json(env: dict[str, str]) -> str:
    """Serialise configs/tenants.yaml (or TENANTS_JSON) for the Space."""
    if env.get("TENANTS_JSON"):
        return env["TENANTS_JSON"]
    path = Path(env.get("TENANTS_FILE") or ROOT / "configs" / "tenants.yaml")
    records = yaml.safe_load(path.read_text(encoding="utf-8"))
    usable = [
        t
        for t in records
        if any("REPLACE_ME" not in str(k.get("argon2_hash", "")) for k in t.get("keys", []))
    ]
    if not usable:
        raise SystemExit("no tenant with a real API key; run scripts/make_api_key.py first")
    return json.dumps(usable, separators=(",", ":"))


def space_host(space: str) -> str:
    return "https://" + space.replace("/", "-").lower() + ".hf.space"


# --------------------------------------------------------------------------- #
# steps
# --------------------------------------------------------------------------- #


def step_verify(env: dict[str, str]) -> str:
    from huggingface_hub import HfApi

    say("1. verify accounts")
    user = str(HfApi(token=env["HF_TOKEN"]).whoami()["name"])
    sys.stdout.write(f"   Hugging Face token works (user {user})\n")
    r = httpx.get(
        env["QDRANT_URL"].rstrip("/") + "/collections",
        headers={"api-key": env["QDRANT_API_KEY"]},
        timeout=15,
    )
    if r.status_code != 200:
        raise SystemExit(f"   cloud Qdrant refused: HTTP {r.status_code} {r.text[:120]}")
    names = [c["name"] for c in r.json().get("result", {}).get("collections", [])]
    sys.stdout.write(f"   cloud Qdrant reachable; collections: {names or 'none yet'}\n")
    return user


def step_repos(env: dict[str, str]) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=env["HF_TOKEN"])
    say("2. repositories")
    api.create_repo(env["HF_DATASET_REPO"], repo_type="dataset", private=True, exist_ok=True)
    sys.stdout.write(f"   dataset {env['HF_DATASET_REPO']} (private) ready\n")
    api.create_repo(
        env["HF_SPACE"], repo_type="space", space_sdk="docker", private=False, exist_ok=True
    )
    sys.stdout.write(f"   space {env['HF_SPACE']} (docker, public) ready\n")


def step_artifacts(env: dict[str, str]) -> None:
    from huggingface_hub import HfApi

    say("3. artifacts -> dataset repo", f"from {STORE_ROOT}")
    api = HfApi(token=env["HF_TOKEN"])
    for prefix in SPACE_PREFIXES:
        folder = STORE_ROOT / prefix
        if not folder.exists():
            sys.stdout.write(f"   {prefix}/: nothing local, skipped\n")
            continue
        count = sum(1 for p in folder.rglob("*") if p.is_file())
        api.upload_folder(
            folder_path=str(folder),
            path_in_repo=prefix,
            repo_id=env["HF_DATASET_REPO"],
            repo_type="dataset",
            commit_message=f"deploy: sync {prefix}/ from local store",
        )
        sys.stdout.write(f"   {prefix}/: {count} file(s) uploaded\n")


def step_vectors(env: dict[str, str]) -> None:
    say("4. cloud Qdrant: create collection and replay artifacts")
    child_env = dict(os.environ)
    child_env.update(
        {
            "QDRANT_URL": env["QDRANT_URL"],
            "QDRANT_API_KEY": env["QDRANT_API_KEY"],
            "VECTOR_BACKEND": "qdrant",
            "ARTIFACT_BACKEND": "local",
            "PYTHONIOENCODING": "utf-8",
        }
    )
    for args in (
        [sys.executable, "scripts/bootstrap_store.py"],
        [sys.executable, "scripts/upsert_artifacts.py", "--tenant", "all", "--verify"],
    ):
        sys.stdout.write(f"   $ {' '.join(args[1:])}\n")
        result = subprocess.run(args, cwd=ROOT, env=child_env, text=True, capture_output=True)
        tail = "\n".join(
            line for line in (result.stdout + result.stderr).splitlines() if '"logger"' not in line
        )[-1500:]
        sys.stdout.write("   " + tail.replace("\n", "\n   ") + "\n")
        if result.returncode != 0:
            raise SystemExit(f"   {args[1]} failed with exit code {result.returncode}")


def step_secrets(env: dict[str, str]) -> None:
    from huggingface_hub import HfApi

    say("5. Space secrets and variables")
    api = HfApi(token=env["HF_TOKEN"])
    space = env["HF_SPACE"]
    webhook = env.get("HF_WEBHOOK_SECRET") or secrets.token_urlsafe(32)
    if not env.get("HF_WEBHOOK_SECRET"):
        with DEPLOY_ENV.open("a", encoding="utf-8") as fh:
            fh.write(f"\nHF_WEBHOOK_SECRET={webhook}\n")
        sys.stdout.write("   generated HF_WEBHOOK_SECRET and appended it to .env.deploy\n")
    secret_values = {
        "QDRANT_URL": env["QDRANT_URL"],
        "QDRANT_API_KEY": env["QDRANT_API_KEY"],
        "GROQ_API_KEY": env.get("GROQ_API_KEY", ""),
        "GEMINI_API_KEY": env.get("GEMINI_API_KEY", ""),
        "HF_TOKEN": env["HF_TOKEN"],
        "HF_WEBHOOK_SECRET": webhook,
        "TENANTS_JSON": tenants_json(env),
    }
    for key, value in secret_values.items():
        if value:
            api.add_space_secret(space, key, value)
    sys.stdout.write(f"   secrets set: {', '.join(k for k, v in secret_values.items() if v)}\n")
    variables = {
        "APP_ENV": "prod",
        "ALLOW_DEV_ANON": "false",
        "ARTIFACT_BACKEND": "hf",
        "HF_DATASET_REPO": env["HF_DATASET_REPO"],
        "VECTOR_BACKEND": "qdrant",
        "EMBED_BACKEND": "fastembed",
        "RETRIEVAL_GATE_TAU": env.get("RETRIEVAL_GATE_TAU", "0.05"),
        "LOG_LEVEL": env.get("LOG_LEVEL", "INFO"),
    }
    for key, value in variables.items():
        api.add_space_variable(space, key, value)
    sys.stdout.write(f"   variables set: {', '.join(variables)}\n")


def step_push(env: dict[str, str]) -> None:
    """Push HEAD to the Space with SPACE_README.md as README.md, via a temporary clone."""
    say("6. push the tree to the Space")
    head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True, capture_output=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True, capture_output=True
    ).stdout.strip()
    if dirty:
        sys.stdout.write("   NOTE: working tree has uncommitted changes; deploying HEAD only\n")
    with tempfile.TemporaryDirectory() as tmp:
        clone = Path(tmp) / "space"
        subprocess.run(
            ["git", "clone", "-q", "--no-local", str(ROOT), str(clone)], check=True, text=True
        )
        shutil.copy(ROOT / "SPACE_README.md", clone / "README.md")
        git = ["git", "-C", str(clone)]
        subprocess.run([*git, "config", "user.name", "deploy"], check=True)
        subprocess.run([*git, "config", "user.email", "deploy@local"], check=True)
        subprocess.run([*git, "add", "-A"], check=True)
        subprocess.run(
            [*git, "commit", "-q", "-m", f"deploy: {head} with Space README front matter"],
            check=True,
        )
        remote = f"https://user:{env['HF_TOKEN']}@huggingface.co/spaces/{env['HF_SPACE']}"
        result = subprocess.run(
            [*git, "push", "--force", remote, "HEAD:main"], text=True, capture_output=True
        )
        if result.returncode != 0:
            scrubbed = (result.stderr or result.stdout).replace(env["HF_TOKEN"], "***")
            raise SystemExit(f"   push failed:\n{scrubbed[-800:]}")
    sys.stdout.write(f"   pushed {head} to {env['HF_SPACE']}\n")


def step_wait(env: dict[str, str], timeout_s: int) -> None:
    host = space_host(env["HF_SPACE"])
    say("7. wait for the Space", f"{host}/healthz (first build bakes the models; ~10-15 min)")
    deadline = time.monotonic() + timeout_s
    live = False
    while time.monotonic() < deadline:
        try:
            if httpx.get(host + "/healthz", timeout=20).status_code == 200:
                live = True
                break
        except httpx.HTTPError:
            pass
        sys.stdout.write(".")
        sys.stdout.flush()
        time.sleep(15)
    sys.stdout.write("\n")
    if not live:
        raise SystemExit("   Space did not become healthy; open its Logs tab in the Space UI")
    sys.stdout.write("   /healthz 200\n")
    for _ in range(40):
        try:
            r = httpx.get(host + "/readyz", timeout=20)
            if r.status_code == 200 and r.json().get("ready"):
                body = r.json()
                sys.stdout.write(f"   /readyz ready; detail={json.dumps(body.get('detail'))}\n")
                break
        except httpx.HTTPError:
            pass
        time.sleep(15)
    else:
        sys.stdout.write("   /readyz not ready yet; models may still be warming\n")
    sys.stdout.write(
        "\nNext:\n"
        f"  curl -sS {host}/v1/meta\n"
        f"  python scripts/check_setup.py   (with QDRANT_URL set to the cloud cluster)\n"
        f"  register the webhook: {host}/v1/internal/hf-webhook with X-Webhook-Secret\n"
    )


# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--plan", action="store_true", help="print the plan and exit")
    parser.add_argument("--skip-store", action="store_true", help="skip artifact and vector sync")
    parser.add_argument("--skip-push", action="store_true", help="only sync store and secrets")
    parser.add_argument("--wait", type=int, default=1200, help="seconds to wait for /healthz")
    args = parser.parse_args()

    env = load_deploy_env()
    missing = check_env(env)
    cloud = bool(env.get("QDRANT_URL")) and "localhost" not in env.get("QDRANT_URL", "")
    sys.stdout.write(
        f"space={env.get('HF_SPACE', '?')}  dataset={env.get('HF_DATASET_REPO', '?')}  "
        f"qdrant={'cloud' if cloud else '?'}\n"
    )
    if missing:
        sys.stdout.write("missing: " + ", ".join(missing) + "\n")
        sys.stdout.write("fill .env.deploy (see .env.deploy.example) and rerun\n")
        return 2
    if args.plan:
        sys.stdout.write(
            "plan: verify -> repos -> artifacts -> vectors -> secrets -> push -> wait\n"
            f"      {'skipping store sync' if args.skip_store else 'store sync on'}; "
            f"{'no push' if args.skip_push else 'push on'}\n"
        )
        return 0

    step_verify(env)
    step_repos(env)
    if not args.skip_store:
        step_artifacts(env)
        step_vectors(env)
    step_secrets(env)
    if not args.skip_push:
        step_push(env)
        step_wait(env, args.wait)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
