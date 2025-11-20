"""
Prompt dispatcher that selects a model-specific template.
"""
from __future__ import annotations
from typing import List, Dict

from src.prompt.gpt_oss_prompt import build_prompt as build_gpt_oss_prompt
from src.prompt.solar_prompt import build_prompt as build_solar_prompt
from src.prompt.eeve_prompt import build_prompt as build_eeve_prompt


def get_prompt(model_name: str | None, data: Dict) -> List[Dict]:
    """
    Resolve the proper prompt builder for the requested model.
    """ 
    if model_name == "gpt-oss-20b":
        return build_gpt_oss_prompt(data)
    elif model_name == "solar-10.7b":
        return build_solar_prompt(data)
    elif model_name == "eeve-10.8b":
        return build_eeve_prompt(data)
    else:
        raise ValueError(f"Not supported model for prompt building: {model_name}")


__all__ = ["get_prompt"]
