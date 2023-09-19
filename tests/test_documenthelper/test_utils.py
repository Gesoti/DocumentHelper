"""Tests for the documenthelper.utils module."""
import typing  # pylint: disable=unused-import
import pytest  # pylint: disable=unused-import
from documenthelper.utils import load_llama_model_configs


class TestUtils:
    """Test class for all utils tests."""

    def test_load_llama_model_configs_correct_keys(self) -> None:
        """Testing the if the configs have the correct top level keys we were expecting."""
        expected_keys = [
            "llama_cpp_shared_configs",
            "llama_embeddings_configs",
            "llama_llm_configs",
        ]
        configs = load_llama_model_configs()
        assert all([k in configs for k in expected_keys]), "Missing keys from config!"
