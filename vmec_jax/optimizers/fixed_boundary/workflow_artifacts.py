"""Result and artifact helpers for fixed-boundary optimization workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def result_timing_summary(result: dict, *, history: dict | None = None) -> dict[str, object]:
    """Extract timing and optimizer call counts from a raw optimizer result."""

    hist = dict(result.get("_history_dump", {}) if history is None else history)
    return {
        "total_wall_time_s": hist.get("total_wall_time_s"),
        "nfev": hist.get("nfev", result.get("nfev")),
        "njev": hist.get("njev", result.get("njev")),
        "nit": hist.get("nit", result.get("nit")),
    }


@dataclass(frozen=True)
class FixedBoundaryOptimizationResult:
    """Result returned by fixed-boundary objective optimization workflows."""

    stage_records: list[tuple[int, object, np.ndarray, dict]]
    final_optimizer: object
    final_result: dict
    stage_modes: list[int]

    @property
    def initial_stage(self) -> tuple[int, object, np.ndarray, dict]:
        """First mode-continuation stage record."""

        return self.stage_records[0]

    @property
    def final_stage(self) -> tuple[int, object, np.ndarray, dict]:
        """Last mode-continuation stage record."""

        return self.stage_records[-1]

    @property
    def initial_optimizer(self):
        """Optimizer object for the first stage."""

        return self.initial_stage[1]

    @property
    def initial_params(self) -> np.ndarray:
        """Initial boundary parameter vector for the first stage."""

        return np.asarray(self.initial_stage[2], dtype=float)

    @property
    def initial_result(self) -> dict:
        """Raw optimizer result dictionary for the first stage."""

        return self.initial_stage[3]

    @property
    def initial_state(self):
        """Initial VMEC state if the optimizer stored one."""

        return self.initial_result.get("_state_initial")

    @property
    def history(self) -> dict:
        """Final optimizer history dictionary written by ``save_history``."""

        return self.final_result.get("_history_dump", {})

    @property
    def history_entries(self) -> tuple[dict, ...]:
        """Per-callback objective samples from the full solve."""

        return tuple(self.history.get("history", ()))

    @property
    def stage_histories(self) -> tuple[dict, ...]:
        """Per-stage history dictionaries in mode-continuation order."""

        return tuple(result.get("_history_dump", {}) for _mode, _optimizer, _params0, result in self.stage_records)

    @property
    def stage_results(self) -> tuple[dict, ...]:
        """Raw optimizer result dictionaries in mode-continuation order."""

        return tuple(result for _mode, _optimizer, _params0, result in self.stage_records)

    @property
    def stage_optimizers(self) -> tuple[object, ...]:
        """Optimizer objects for each mode-continuation stage."""

        return tuple(optimizer for _mode, optimizer, _params0, _result in self.stage_records)

    @property
    def stage_initial_params(self) -> tuple[np.ndarray, ...]:
        """Initial boundary-parameter vectors for each stage."""

        return tuple(np.asarray(params0, dtype=float) for _mode, _optimizer, params0, _result in self.stage_records)

    @property
    def objective_history(self) -> np.ndarray:
        """Objective values over full-solve callbacks as a NumPy array."""

        return np.asarray([entry.get("objective", np.nan) for entry in self.history_entries], dtype=float)

    @property
    def final_params(self) -> np.ndarray:
        """Optimized boundary parameter vector for the final stage."""

        return np.asarray(self.final_result["x"], dtype=float)

    @property
    def final_state(self):
        """Final VMEC state if the optimizer stored one."""

        return self.final_result.get("_state_final")

    @property
    def stage_timing_summaries(self) -> tuple[dict[str, object], ...]:
        """Small timing/iteration summaries for each stage."""

        summaries = []
        for mode, _optimizer, _params0, result in self.stage_records:
            summary = result_timing_summary(result)
            summary["mode"] = int(mode)
            summaries.append(summary)
        return tuple(summaries)

    @property
    def timing_summary(self) -> dict[str, object]:
        """Small timing/iteration summary for reports and examples."""

        summary = result_timing_summary(self.final_result, history=self.history)
        summary["stages"] = self.stage_timing_summaries
        return summary

    @property
    def summary(self) -> dict[str, object]:
        """Compact result summary for example reports and notebooks."""

        history = self.history
        return {
            "stage_modes": tuple(int(mode) for mode in self.stage_modes),
            "objective_initial": history.get("objective_initial"),
            "objective_final": history.get("objective_final"),
            "aspect_final": history.get("aspect_final"),
            "iota_final": history.get("iota_final"),
            "field_objective_final": history.get("qs_final"),
            "timing": self.timing_summary,
        }


@dataclass(frozen=True)
class OptimizationOutputPaths:
    """Canonical files written by fixed-boundary optimization examples."""

    initial_input: Path
    final_input: Path
    initial_wout: Path
    final_wout: Path
    history: Path

    def as_dict(self) -> dict[str, Path]:
        """Return path names in the same order used by the example reports."""

        return {
            "initial_input": self.initial_input,
            "final_input": self.final_input,
            "initial_wout": self.initial_wout,
            "final_wout": self.final_wout,
            "history": self.history,
        }


def optimization_output_paths(output_dir: str | Path) -> OptimizationOutputPaths:
    """Return the canonical final-artifact paths for an optimization run."""

    output_dir = Path(output_dir)
    return OptimizationOutputPaths(
        initial_input=output_dir / "input.initial",
        final_input=output_dir / "input.final",
        initial_wout=output_dir / "wout_initial.nc",
        final_wout=output_dir / "wout_final.nc",
        history=output_dir / "history.json",
    )


def save_optimization_result(
    result: FixedBoundaryOptimizationResult,
    *,
    output_dir: str | Path | None = None,
    paths: OptimizationOutputPaths | None = None,
) -> OptimizationOutputPaths:
    """Save initial/final inputs, wouts, and history from a solve result."""

    if paths is None:
        if output_dir is None:
            raise ValueError("Either output_dir or paths must be provided.")
        paths = optimization_output_paths(output_dir)
    for path in paths.as_dict().values():
        path.parent.mkdir(parents=True, exist_ok=True)

    result.initial_optimizer.save_input(paths.initial_input, result.initial_params)
    result.initial_optimizer.save_wout(
        paths.initial_wout,
        result.initial_params,
        state=result.initial_state,
    )
    result.final_optimizer.save_input(paths.final_input, result.final_params)
    result.final_optimizer.save_wout(
        paths.final_wout,
        result.final_params,
        state=result.final_state,
    )
    result.final_optimizer.save_history(paths.history, result.final_result)
    return paths

