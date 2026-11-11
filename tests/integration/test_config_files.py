"""Tests of the configuration files: knowledge base and models of each role."""

import json
import pathlib

import pydantic
import pytest

from agent.models import model_config
from shared import kb_config

pytestmark = pytest.mark.integration

TITAN_MODEL_ID = "amazon.titan-embed-text-v2:0"


def _bedrock_model(**overrides) -> dict:
    return {
        "provider": "bedrock",
        "model_id": "claude",
        "temperature": 0.0,
        "params": {"region_name": "eu-west-1"},
    } | overrides


def _write_models(tmp_path: pathlib.Path, roles: dict) -> pathlib.Path:
    path = tmp_path / "models.json"
    path.write_text(json.dumps(roles), encoding="utf-8")
    return path


class TestModelCatalog:
    """The catalog needs every role and valid providers."""

    def test_missing_role_raises(self, tmp_path):
        """Test: catalog without the judge role.
        Expected: ValueError naming the role.
        """
        # Arrange
        roles = {
            role: {"description": role, "models": [_bedrock_model()]}
            for role in ("triage", "scope")
        }

        # Act / Assert
        with pytest.raises(ValueError, match="judge"):
            model_config.ModelCatalog.from_file(_write_models(tmp_path, roles))

    def test_bedrock_model_without_region_raises(self, tmp_path):
        """Test: Bedrock model without region_name.
        Expected: ValueError naming the missing param.
        """
        # Arrange
        roles = {
            role: {"description": role, "models": [_bedrock_model(params={})]}
            for role in ("triage", "scope", "judge")
        }

        # Act / Assert
        with pytest.raises(ValueError, match="region_name"):
            model_config.ModelCatalog.from_file(_write_models(tmp_path, roles))


class TestKnowledgeBaseConfig:
    """The configuration validates embedder, chunking and retrieval."""

    @staticmethod
    def _write(tmp_path: pathlib.Path, *, dimensions: int) -> pathlib.Path:
        path = tmp_path / "knowledge_base.json"
        path.write_text(
            json.dumps(
                {
                    "embedder": {
                        "model_id": TITAN_MODEL_ID,
                        "region_name": "eu-west-1",
                        "dimensions": dimensions,
                        "max_concurrency": 4,
                    },
                    "chunking": {"max_chars": 1200, "overlap_chars": 150},
                    "retrieval": {"top_k": 5, "min_score": 0.2},
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_unsupported_dimensions_are_invalid(self, tmp_path):
        """Test: vector size that Titan does not support.
        Expected: validation error.
        """
        # Act / Assert
        with pytest.raises(pydantic.ValidationError, match="dimensions"):
            kb_config.KnowledgeBaseConfig.from_file(
                self._write(tmp_path, dimensions=300)
            )
