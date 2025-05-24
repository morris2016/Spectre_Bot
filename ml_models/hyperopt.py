#!/usr/bin/env python3
"""Unified hyperparameter optimization service."""

from __future__ import annotations

from typing import Dict, Any

import numpy as np
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK


class HyperOptService:
    """Run hyperparameter optimization using Hyperopt."""

    def __init__(self, max_evals: int = 50):
        self.max_evals = max_evals

    def optimize(self, objective_fn, search_space: Dict[str, Any]) -> Dict[str, Any]:
        trials = Trials()
        best_params = fmin(
            fn=objective_fn,
            space=search_space,
            algo=tpe.suggest,
            max_evals=self.max_evals,
            trials=trials,
        )
        # Convert ints
        final_params: Dict[str, Any] = {}
        for k, v in best_params.items():
            if isinstance(v, float) and k.startswith("n_"):
                final_params[k] = int(v)
            else:
                final_params[k] = v
        return final_params
