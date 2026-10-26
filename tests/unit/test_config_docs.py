"""Tests that keep configuration documentation honest.

Documentation about configuration goes stale silently: a new setting is added,
``.env.example`` is not, and the next person deploying discovers the gap at
runtime. These checks make that a build failure instead.
"""

from __future__ import annotations

import pathlib
import re

from emrag.settings import CONFIG_DIR, Settings

REPO = pathlib.Path(__file__).resolve().parents[2]
WORKFLOWS = REPO / ".github" / "workflows"
ENV_EXAMPLE = REPO / ".env.example"
DEPLOY_RUNBOOK = REPO / "docs" / "runbooks" / "deploy.md"

# Settings with no business being in a sample env file.
INTERNAL_SETTINGS = {"TENANTS_FILE"}


def _documented_env_keys() -> set[str]:
    text = ENV_EXAMPLE.read_text(encoding="utf-8")
    return set(re.findall(r"^([A-Z][A-Z0-9_]*)=", text, re.MULTILINE))


def _workflow_text() -> str:
    return "".join(p.read_text(encoding="utf-8") for p in sorted(WORKFLOWS.glob("*.yml")))


class TestEnvExample:
    """``.env.example`` is the contract for what a deployment must set."""

    def test_every_setting_is_documented(self) -> None:
        declared = {name.upper() for name in Settings.model_fields} - INTERNAL_SETTINGS
        missing = sorted(declared - _documented_env_keys())
        assert not missing, f".env.example is missing: {missing}"

    def test_no_stale_entries(self) -> None:
        declared = {name.upper() for name in Settings.model_fields}
        stale = sorted(_documented_env_keys() - declared)
        assert not stale, f".env.example documents settings that no longer exist: {stale}"

    def test_contains_no_real_looking_values(self) -> None:
        # A sample file people copy must never carry a usable credential.
        text = ENV_EXAMPLE.read_text(encoding="utf-8")
        for pattern in (
            r"gsk_[A-Za-z0-9]{20,}",
            r"AIza[0-9A-Za-z_\-]{30,}",
            r"hf_[A-Za-z0-9]{30,}",
        ):
            assert not re.search(pattern, text), f"credential-shaped value matching {pattern}"


class TestDeployRunbook:
    """The deploy runbook must list every secret a workflow needs."""

    def test_documents_every_workflow_secret(self) -> None:
        used = set(re.findall(r"secrets\.([A-Z0-9_]+)", _workflow_text())) - {"GITHUB_TOKEN"}
        runbook = DEPLOY_RUNBOOK.read_text(encoding="utf-8")
        missing = sorted(s for s in used if s not in runbook)
        assert not missing, f"deploy runbook does not mention secrets: {missing}"

    def test_documents_every_workflow_variable(self) -> None:
        used = set(re.findall(r"vars\.([A-Z0-9_]+)", _workflow_text()))
        runbook = DEPLOY_RUNBOOK.read_text(encoding="utf-8")
        missing = sorted(v for v in used if v not in runbook)
        assert not missing, f"deploy runbook does not mention variables: {missing}"


class TestWorkflowReferences:
    """Workflows must not reference scripts or make targets that do not exist."""

    def test_referenced_scripts_exist(self) -> None:
        referenced = set(re.findall(r"scripts/[a-z_]+\.py", _workflow_text()))
        missing = sorted(s for s in referenced if not (REPO / s).is_file())
        assert not missing, f"workflows reference missing scripts: {missing}"

    def test_readme_make_targets_exist(self) -> None:
        readme = (REPO / "README.md").read_text(encoding="utf-8")
        makefile = (REPO / "Makefile").read_text(encoding="utf-8")
        targets = set(re.findall(r"^make ([a-z-]+)", readme, re.MULTILINE))
        missing = sorted(t for t in targets if not re.search(rf"^{t}:", makefile, re.MULTILINE))
        assert not missing, f"README references missing make targets: {missing}"


class TestTuningConfig:
    """Sections the code reads from ``configs/default.yaml`` must be present."""

    def test_required_sections_exist(self) -> None:
        import yaml

        data = yaml.safe_load((CONFIG_DIR / "default.yaml").read_text(encoding="utf-8"))
        for section in ("chunking", "retrieval", "generation", "figures", "ocr"):
            assert section in data, f"configs/default.yaml is missing section {section}"

    def test_chunking_keys_match_the_dataclass(self) -> None:
        import yaml

        from emrag.ingestion.chunking import ChunkingConfig

        data = yaml.safe_load((CONFIG_DIR / "default.yaml").read_text(encoding="utf-8"))
        known = set(ChunkingConfig.__dataclass_fields__)
        unknown = sorted(set(data["chunking"]) - known)
        assert not unknown, f"configs/default.yaml chunking has unused keys: {unknown}"

    def test_tenants_example_has_no_usable_hash(self) -> None:
        text = (CONFIG_DIR / "tenants.example.yaml").read_text(encoding="utf-8")
        assert "REPLACE_ME" in text


class TestWarmUpGate:
    """Background ingestion waits for start-up rather than competing with it."""

    async def test_wait_returns_immediately_once_marked(self) -> None:
        import asyncio
        import time

        from emrag.api.deps import AppState

        state = AppState.__new__(AppState)
        state.warm = False
        state.warm_event = asyncio.Event()
        state.mark_warm(ready=True)

        started = time.perf_counter()
        assert await state.wait_until_warm(timeout_s=5.0) is True
        assert time.perf_counter() - started < 0.5

    async def test_wait_reports_failed_start_up(self) -> None:
        import asyncio

        from emrag.api.deps import AppState

        state = AppState.__new__(AppState)
        state.warm = False
        state.warm_event = asyncio.Event()
        # A warm-up that failed must still release waiters, so a queued upload
        # fails fast instead of sitting behind something that will never finish.
        state.mark_warm(ready=False)
        assert await state.wait_until_warm(timeout_s=5.0) is False

    async def test_wait_times_out_rather_than_hanging(self) -> None:
        import asyncio

        from emrag.api.deps import AppState

        state = AppState.__new__(AppState)
        state.warm = False
        state.warm_event = asyncio.Event()
        assert await state.wait_until_warm(timeout_s=0.05) is False


class TestSecretFilesAreIgnored:
    """Files that hold credentials must be ignored by git.

    configs/tenants.yaml was documented as ignored but was not, and a careless
    `git add -A` would have committed the tenant records. The ignore rules are
    checked by asking git itself, so this cannot drift from the real behaviour.
    """

    @pytest.mark.parametrize("path", [".env", ".env.local", "configs/tenants.yaml"])
    def test_git_ignores(self, path: str) -> None:
        import shutil
        import subprocess

        if shutil.which("git") is None or not (REPO / ".git").exists():
            pytest.skip("not a git checkout")
        result = subprocess.run(
            ["git", "check-ignore", "-q", path], cwd=REPO, capture_output=True, check=False
        )
        assert result.returncode == 0, f"{path} is not ignored by git"
