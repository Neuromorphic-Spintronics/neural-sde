"""Sanity checks for example notebooks.

These tests import the marimo notebooks to ensure they are importable,
define an `app` object, and do not rely on sys.path hacks.
"""

import importlib
import types
import pytest


@pytest.mark.sanity
def test_duffing_notebook_imports() -> None:
    mod = importlib.import_module("examples.duffing_oscillator_notebook")
    assert isinstance(mod, types.ModuleType)
    assert hasattr(mod, "app"), "marimo App `app` should be defined"


@pytest.mark.sanity
def test_nanorings_notebook_imports() -> None:
    mod = importlib.import_module("examples.nanorings_digital_twin")
    assert isinstance(mod, types.ModuleType)
    assert hasattr(mod, "app"), "marimo App `app` should be defined"

