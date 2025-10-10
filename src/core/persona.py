from __future__ import annotations
"""Persona loading utilities."""
from pathlib import Path
import json
from typing import Union

from src.core.schemas import Persona

_DEF_PATHS = [
    Path("data/mock/persona.json"),
    Path("sample_persona.json"),
]

def load_persona(path: Union[str, Path] | None = None) -> Persona:
    if path is None:
        for p in _DEF_PATHS:
            if p.exists():
                path = p
                break
    if path is None:
        raise FileNotFoundError("Persona file not found in default paths.")
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return Persona(**data)

__all__ = ["load_persona"]
