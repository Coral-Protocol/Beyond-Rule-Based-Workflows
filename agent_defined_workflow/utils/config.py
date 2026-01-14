"""Configuration file for model settings"""
import os

from camel.models import ModelFactory, BaseModelBackend
from camel.types import ModelType, ModelPlatformType

# Model Configuration 
# for more information on the models, see https://github.com/camel-ai/camel/blob/master/camel/types/enums.py

# Model Settings
MODEL_CONFIG = {
    "temperature": 0,
    "frequency_penalty": 0,
    "top_p": 0.99,
}

def get_model() -> BaseModelBackend:
    """Get the model for general tasks."""
    return ModelFactory.create(
            model_platform="openrouter",
            model_type="x-ai/grok-4.1-fast",
            url = "https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),  # Set your API key if needed
            model_config_dict=MODEL_CONFIG
        )

def get_process_model() -> BaseModelBackend:
    """Get the model for general tasks."""
    return ModelFactory.create(
            model_platform="openrouter",
            model_type="openai/gpt-4.1-mini",
            url = "https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),  # Set your API key if needed
            model_config_dict=MODEL_CONFIG
        )

def get_worker_model() -> BaseModelBackend:
    """Get the model for worker agent."""
    return ModelFactory.create(
            model_platform="openrouter",
            model_type="x-ai/grok-4.1-fast",
            url = "https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),  # Set your API key if needed
            model_config_dict=MODEL_CONFIG
        )

def get_reasoning_worker_model() -> BaseModelBackend:
    """Get the model for worker agent."""
    return ModelFactory.create(
            model_platform="openrouter",
            model_type="x-ai/grok-4.1-fast",
            url = "https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),  # Set your API key if needed
            model_config_dict=MODEL_CONFIG
        )

def get_web_model() -> BaseModelBackend:
    """Get the model for worker agent."""
    return ModelFactory.create(
            model_platform="openrouter",
            model_type="x-ai/grok-4.1-fast",
            url = "https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),  # Set your API key if needed
            model_config_dict=MODEL_CONFIG
        )

def get_web_planning_model() -> BaseModelBackend:
    """Get the model for worker agent."""
    return ModelFactory.create(
            model_platform="openrouter",
            model_type="x-ai/grok-4.1-fast",
            url = "https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),  # Set your API key if needed
            model_config_dict=MODEL_CONFIG
        )

def get_image_model() -> BaseModelBackend:
    """Get the model for worker agent."""
    return ModelFactory.create(
            model_platform="openrouter",
            model_type="x-ai/grok-4.1-fast",
            url = "https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),  # Set your API key if needed
            model_config_dict=MODEL_CONFIG
        )

def get_audio_model() -> BaseModelBackend:
    """Get the model for worker agent."""
    return ModelFactory.create(
            model_platform="openrouter",
            model_type="x-ai/grok-4.1-fast",
            url = "https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),  # Set your API key if needed
            model_config_dict=MODEL_CONFIG
        )








