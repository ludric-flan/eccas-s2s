"""
Figures written to a file are drawn with pyplot's interactive mode off.

In a Jupyter kernel the inline backend keeps a reference to **every** figure
drawn during a cell, so that it can show them when the cell ends — even after
``plt.close``. A cell that draws hundreds of maps (OP_07 to OP_10) therefore
holds them all in memory: 26 calibrated maps took 1.9 GB against 0.7 GB in a
script, and the 780 multi-model maps of the cycle 202611 were killed by the
system at 22 GB (20/11/2026). A figure written to ``output_path`` is saved and
closed, nobody needs it on screen, so it is drawn with the interactive mode off
and the inline backend never sees it. A figure drawn without ``output_path`` —
to look at it in a notebook — is left alone.
"""
from __future__ import annotations

import inspect
from functools import wraps


def offscreen_when_saved(func):
    """Decorator: run ``func`` under ``plt.ioff()`` when it is given an ``output_path``."""
    signature = inspect.signature(func)

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            target = signature.bind_partial(*args, **kwargs).arguments.get("output_path")
        except TypeError:
            target = kwargs.get("output_path")
        if not target:
            return func(*args, **kwargs)
        import matplotlib.pyplot as plt

        with plt.ioff():
            return func(*args, **kwargs)

    return wrapper
