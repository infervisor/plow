"""Opt-in forward capture for unmodified vLLM worker processes.

Set ``PLOW_VLLM_CAPTURE_CONFIG`` to a JSON config path and put this directory
first on ``PYTHONPATH``.  With the variable unset this module is a no-op.
"""

import os


if os.environ.get("PLOW_VLLM_CAPTURE_CONFIG"):
    from vllm_forward_capture import install_from_env

    install_from_env()
