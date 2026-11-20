"""Verify the installed SDK interface without loading a live provider or remote cost map."""

import os
from importlib.metadata import version
from inspect import Parameter, signature

os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "true"

from litellm import acompletion


def test_locked_litellm_version_has_async_completion_adapter_surface():
    parameters = signature(acompletion).parameters
    assert {"model", "messages", "timeout", "api_key"} <= parameters.keys()
    assert any(parameter.kind is Parameter.VAR_KEYWORD for parameter in parameters.values())
    assert version("litellm")
