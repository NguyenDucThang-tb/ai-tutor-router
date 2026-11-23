from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "normalize_kaggle_notebook.py"
_SPEC = importlib.util.spec_from_file_location("normalize_kaggle_notebook", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
normalizer = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(normalizer)


def _fixture(path: Path) -> dict:
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": 7,
                "id": "executed-cell",
                "metadata": {"tags": ["gpu-evidence"]},
                "outputs": [
                    {
                        "data": {
                            "text/plain": ["{'accuracy': 0.9}\n"],
                            "image/png": ["evidence-bytes"],
                        },
                        "execution_count": 7,
                        "metadata": {"width": 100},
                        "output_type": "execute_result",
                    }
                ],
                "source": ["value=1\n", "print(value)\n"],
            },
            {
                "cell_type": "markdown",
                "id": "markdown-cell",
                "metadata": {"collapsed": False},
                "source": ["# Saved result\n", "Historical output stays attached."],
            },
        ],
        "metadata": {"kaggle": {"accelerator": "nvidiaTeslaT4"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(notebook), encoding="utf-8")
    return notebook


def test_notebook_normalization_preserves_evidence_and_is_idempotent(
    tmp_path: Path,
) -> None:
    path = tmp_path / "pilot.ipynb"
    before = _fixture(path)

    assert normalizer.normalize_notebook(path) is True
    once_bytes = path.read_bytes()
    after = json.loads(once_bytes)

    assert after["metadata"] == before["metadata"]
    assert after["nbformat"] == before["nbformat"]
    assert after["nbformat_minor"] == before["nbformat_minor"]
    assert "".join(after["cells"][0]["source"]) == "value = 1\nprint(value)"
    assert after["cells"][0]["execution_count"] == before["cells"][0]["execution_count"]
    assert after["cells"][0]["outputs"] == before["cells"][0]["outputs"]
    assert after["cells"][0]["metadata"] == before["cells"][0]["metadata"]
    assert after["cells"][1] == before["cells"][1]

    assert normalizer.normalize_notebook(path) is False
    assert path.read_bytes() == once_bytes


def test_notebook_normalizer_refuses_to_replace_a_concurrent_edit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "pilot.ipynb"
    _fixture(path)
    original_run = subprocess.run

    def format_then_edit(*args, **kwargs):
        result = original_run(*args, **kwargs)
        path.write_bytes(path.read_bytes() + b" ")
        return result

    monkeypatch.setattr(normalizer.subprocess, "run", format_then_edit)
    with pytest.raises(normalizer.NotebookNormalizationError, match="changed while"):
        normalizer.normalize_notebook(path)

    assert path.read_bytes().endswith(b" ")
