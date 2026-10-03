"""
Test Groq Model Configuration
==============================
Regression test to ensure the chatbot and domain translator use the correct Groq model.
This test verifies that the obsolete model (llama-3.3-70b-versatile) is not used
and that the new model (openai/gpt-oss-120b) is correctly configured.
"""

import os
import pytest


class TestGroqModelConfiguration:
    """Test that Groq model configuration uses the correct model."""

    def test_chatbot_routes_default_model(self):
        """Test that chatbot_routes.py uses the correct default model."""
        # Remove GROQ_MODEL from environment to test default
        if "GROQ_MODEL" in os.environ:
            del os.environ["GROQ_MODEL"]

        # Import after clearing env var to ensure fresh load
        import importlib
        import sys

        # Remove from cache if already imported
        if "webapp.routes.chatbot_routes" in sys.modules:
            del sys.modules["webapp.routes.chatbot_routes"]

        from webapp.routes.chatbot_routes import MODEL_NAME

        # Should default to openai/gpt-oss-120b
        assert MODEL_NAME == "openai/gpt-oss-120b", \
            f"Expected default model to be 'openai/gpt-oss-120b', got '{MODEL_NAME}'"

        # Should NOT be the obsolete model
        assert MODEL_NAME != "llama-3.3-70b-versatile", \
            "Obsolete model 'llama-3.3-70b-versatile' should not be used"

    def test_chatbot_routes_env_override(self):
        """Test that chatbot_routes.py respects GROQ_MODEL environment variable."""
        # Set custom model
        os.environ["GROQ_MODEL"] = "custom-test-model"

        # Reload module
        import sys
        if "webapp.routes.chatbot_routes" in sys.modules:
            del sys.modules["webapp.routes.chatbot_routes"]

        from webapp.routes.chatbot_routes import MODEL_NAME

        # Should use environment variable
        assert MODEL_NAME == "custom-test-model", \
            f"Expected model from env var to be 'custom-test-model', got '{MODEL_NAME}'"

        # Cleanup
        del os.environ["GROQ_MODEL"]

    def test_domain_translator_default_model(self):
        """Test that domain_translator.py uses the correct default model."""
        # Remove GROQ_MODEL from environment to test default
        if "GROQ_MODEL" in os.environ:
            del os.environ["GROQ_MODEL"]

        # Import after clearing env var to ensure fresh load
        import importlib
        import sys

        # Remove from cache if already imported
        if "engine.modules.domain_translator" in sys.modules:
            del sys.modules["engine.modules.domain_translator"]

        from engine.modules.domain_translator import MODEL_NAME

        # Should default to openai/gpt-oss-120b
        assert MODEL_NAME == "openai/gpt-oss-120b", \
            f"Expected default model to be 'openai/gpt-oss-120b', got '{MODEL_NAME}'"

        # Should NOT be the obsolete model
        assert MODEL_NAME != "llama-3.3-70b-versatile", \
            "Obsolete model 'llama-3.3-70b-versatile' should not be used"

    def test_domain_translator_env_override(self):
        """Test that domain_translator.py respects GROQ_MODEL environment variable."""
        # Set custom model
        os.environ["GROQ_MODEL"] = "custom-test-model"

        # Reload module
        import sys
        if "engine.modules.domain_translator" in sys.modules:
            del sys.modules["engine.modules.domain_translator"]

        from engine.modules.domain_translator import MODEL_NAME

        # Should use environment variable
        assert MODEL_NAME == "custom-test-model", \
            f"Expected model from env var to be 'custom-test-model', got '{MODEL_NAME}'"

        # Cleanup
        del os.environ["GROQ_MODEL"]
