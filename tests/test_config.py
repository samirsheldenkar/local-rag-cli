"""Tests for configuration module."""

import os
from unittest.mock import patch

import pytest

from local_rag_cli.config import Settings


class TestSettings:
    """Test suite for Settings configuration."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        settings = Settings()

        assert settings.QDRANT_URL == "http://localhost:6333"
        assert settings.QDRANT_API_KEY is None
        assert settings.LLM_BASE_URL == "http://localhost:1234/v1"
        assert settings.LLM_API_KEY is None
        assert settings.LLM_MODEL == "local-model"
        assert settings.TEXT_EMBEDDING_MODEL == "BAAI/bge-m3"
        assert settings.IMAGE_EMBEDDING_MODEL == "ViT-B/32"
        assert settings.REQUEST_TIMEOUT == 600.0
        assert settings.LOG_LEVEL == "INFO"

    def test_env_var_override(self):
        """Test that environment variables override defaults."""
        with patch.dict(
            os.environ,
            {
                "QDRANT_URL": "http://qdrant.example.com:6333",
                "LLM_MODEL": "custom-model",
                "REQUEST_TIMEOUT": "300.0",
            },
        ):
            settings = Settings()

            assert settings.QDRANT_URL == "http://qdrant.example.com:6333"
            assert settings.LLM_MODEL == "custom-model"
            assert settings.REQUEST_TIMEOUT == 300.0

    def test_api_key_from_env(self):
        """Test that API keys can be loaded from environment."""
        with patch.dict(
            os.environ,
            {
                "QDRANT_API_KEY": "secret-qdrant-key",
                "LLM_API_KEY": "secret-llm-key",
            },
        ):
            settings = Settings()

            assert settings.QDRANT_API_KEY == "secret-qdrant-key"
            assert settings.LLM_API_KEY == "secret-llm-key"

    def test_request_timeout_type(self):
        """Test that REQUEST_TIMEOUT is parsed as float."""
        with patch.dict(os.environ, {"REQUEST_TIMEOUT": "120"}):
            settings = Settings()
            assert isinstance(settings.REQUEST_TIMEOUT, float)
            assert settings.REQUEST_TIMEOUT == 120.0
