"""Tests for the provider catalogue in ``configs/providers.yaml``.

The catalogue is data, not code, so nothing else validates it. These checks exist
because two of its failure modes are silent: a duplicated model inflates the
quota the router believes it has, and a missing tier makes a whole capability
unavailable with no error.
"""

from __future__ import annotations

import pytest
import yaml

from emrag.llm.router import _parse_spec
from emrag.settings import CONFIG_DIR

REQUIRED_TIERS = {"fast", "quality", "rewrite", "repair", "vision", "judge"}
KNOWN_PROVIDERS = {"groq", "gemini"}


@pytest.fixture(scope="module")
def catalogue() -> dict[str, object]:
    return yaml.safe_load((CONFIG_DIR / "providers.yaml").read_text(encoding="utf-8"))


class TestCatalogue:
    """Structural invariants of the model catalogue."""

    def test_every_model_parses_into_a_spec(self, catalogue: dict) -> None:
        specs = [_parse_spec(entry) for entry in catalogue["models"]]
        assert specs
        assert all(spec.key and spec.model for spec in specs)

    def test_model_keys_are_unique(self, catalogue: dict) -> None:
        keys = [entry["key"] for entry in catalogue["models"]]
        assert len(keys) == len(set(keys))

    def test_each_underlying_model_appears_once(self, catalogue: dict) -> None:
        # The quota ledger keys on `key`. Two entries for one provider model
        # would be given two independent budgets and overspend the real one.
        pairs = [(e["provider"], e["model"]) for e in catalogue["models"]]
        duplicates = {p for p in pairs if pairs.count(p) > 1}
        assert not duplicates, f"duplicated provider models: {sorted(duplicates)}"

    def test_every_tier_has_a_model(self, catalogue: dict) -> None:
        covered: set[str] = set()
        for entry in catalogue["models"]:
            covered.update(entry["tiers"])
        missing = REQUIRED_TIERS - covered
        assert not missing, f"no model serves tier(s): {sorted(missing)}"

    def test_vision_tier_models_declare_vision(self, catalogue: dict) -> None:
        for entry in catalogue["models"]:
            if "vision" in entry["tiers"]:
                assert entry["capabilities"]["vision"] is True, entry["key"]

    def test_providers_are_known_and_defined(self, catalogue: dict) -> None:
        defined = set(catalogue["providers"])
        assert defined <= KNOWN_PROVIDERS
        for entry in catalogue["models"]:
            assert entry["provider"] in defined, entry["key"]

    def test_limits_are_positive_or_absent(self, catalogue: dict) -> None:
        for entry in catalogue["models"]:
            for name, value in (entry.get("limits") or {}).items():
                assert value is None or value > 0, f"{entry['key']}.{name} = {value}"

    def test_each_tier_has_more_than_one_model_where_it_matters(self, catalogue: dict) -> None:
        # fast and vision carry the volume; a single model there means one
        # exhausted quota takes the capability down with no fallback.
        counts: dict[str, int] = {}
        for entry in catalogue["models"]:
            for tier in entry["tiers"]:
                counts[tier] = counts.get(tier, 0) + 1
        for tier in ("fast", "quality", "vision", "rewrite"):
            assert counts.get(tier, 0) >= 2, f"tier {tier} has no fallback model"

    def test_routing_section_is_sane(self, catalogue: dict) -> None:
        routing = catalogue["routing"]
        assert 0.0 <= routing["daily_reserve_ratio"] < 1.0
        assert routing["max_attempts_per_tier"] >= 1
        assert routing["request_timeout_s"] > routing["connect_timeout_s"]
