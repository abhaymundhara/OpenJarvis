"""Tests for the Learning Dashboard API endpoints."""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
starlette = pytest.importorskip("starlette")

from fastapi import FastAPI  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

from openjarvis.server.api_routes import learning_router  # noqa: E402


def _make_app() -> FastAPI:
    """Create a minimal FastAPI app with the learning router included."""
    app = FastAPI()
    app.include_router(learning_router)
    return app


def _client() -> TestClient:
    return TestClient(_make_app())


# ---- /v1/learning/stats tests ----


def test_learning_stats_returns_200():
    """GET /v1/learning/stats should return 200."""
    client = _client()
    resp = client.get("/v1/learning/stats")
    assert resp.status_code == 200


def test_learning_stats_has_all_sections():
    """Response must contain grpo, bandit, icl, and skill_discovery sections."""
    client = _client()
    data = client.get("/v1/learning/stats").json()
    for section in ("grpo", "bandit", "icl", "skill_discovery"):
        assert section in data, f"Missing section: {section}"
        assert "available" in data[section], (
            f"Section '{section}' missing 'available' key"
        )


def test_learning_stats_grpo_fields():
    """When GRPO is available, verify expected stat fields are present."""
    client = _client()
    data = client.get("/v1/learning/stats").json()
    grpo = data["grpo"]
    if grpo["available"]:
        assert "total_updates" in grpo
        assert "sample_counts" in grpo
        assert "weight_count" in grpo


def test_learning_stats_icl_fields():
    """When ICL is available, verify expected stat fields are present."""
    client = _client()
    data = client.get("/v1/learning/stats").json()
    icl = data["icl"]
    if icl["available"]:
        assert "example_count" in icl
        assert "example_db_count" in icl
        assert "version" in icl


# ---- /v1/learning/policy tests ----


def test_learning_policy_returns_200():
    """GET /v1/learning/policy should return 200."""
    client = _client()
    resp = client.get("/v1/learning/policy")
    assert resp.status_code == 200


def test_learning_policy_has_expected_keys():
    """Response must include enabled, routing, intelligence, agent, metrics."""
    client = _client()
    data = client.get("/v1/learning/policy").json()
    assert "enabled" in data
    assert "routing" in data
    assert "intelligence" in data
    assert "agent" in data
    assert "metrics" in data


def test_learning_policy_routing_structure():
    """The routing section should contain policy name and min_samples."""
    client = _client()
    data = client.get("/v1/learning/policy").json()
    routing = data["routing"]
    assert "policy" in routing
    assert "min_samples" in routing
    assert isinstance(routing["policy"], str)
    assert isinstance(routing["min_samples"], int)


def test_learning_policy_enabled_is_bool():
    """The enabled field should be a boolean."""
    client = _client()
    data = client.get("/v1/learning/policy").json()
    assert isinstance(data["enabled"], bool)
