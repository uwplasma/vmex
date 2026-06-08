"""Support routines for staged bounded QI optimization.

The public example script keeps the user-facing workflow visible: configure
inputs, construct objective tuples, call ``least_squares_solve``, then save/plot
outputs.  This module holds the reusable seed-preconditioning, stage-promotion,
and checkpoint helpers used by that script and sweep drivers.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

import vmec_jax as vj
from vmec_jax.namelist import InData
from vmec_jax.optimization import boundary_param_names, create_x_scale
from vmec_jax.qi_diagnostics import QISeedSuitabilityTargets, annotate_qi_seed_suitability


def _load_basin_prefilter_tools():
    """Load repo-local diagnostic helpers only when the optional prefilter runs.

    ``tools/`` is intentionally excluded from the wheel, so importing this
    public module must not require those scripts for normal installed-package
    use. The example tree can still use the large-step basin prefilter when run
    from a source checkout.
    """

    try:
        from tools.diagnostics.qi_basin_survey import (
            SurveyTargets,
            generate_basin_candidates,
            rank_candidate_records,
            write_csv,
        )
        from tools.diagnostics.qi_landscape_scan import build_stage as build_diagnostic_stage
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The QI basin prefilter requires vmec_jax to be run from a source checkout "
            "with the repo-local tools/diagnostics scripts available."
        ) from exc
    return SurveyTargets, generate_basin_candidates, rank_candidate_records, write_csv, build_diagnostic_stage


__all__ = [
    "TARGET_HELICITY_SEED_AMPLITUDE",
    "TARGET_HELICITY_SEED_MODE_TERMS",
    "QIOptimizationContext",
    "apply_qi_example_cli_overrides",
    "basin_prefilter_score",
    "boundary_reference_preconditioner_score",
    "boundary_reference_record_is_qi_safe",
    "configure",
    "diagnostic_float",
    "engineering_promotion_score",
    "jsonable",
    "make_basin_prefilter_options",
    "make_qi_optimization_context",
    "materialize_qi_stage_inputs",
    "promotion_score",
    "qi_mirror_objective_for_stage",
    "qi_engineering_constraint_tuples",
    "qi_diagnostics_for_result",
    "qi_diagnostics_for_run",
    "qi_stage_modes",
    "run_basin_prefilter",
    "run_boundary_reference_preconditioner",
    "run_qi_stage_policy",
    "run_target_helicity_seed_preconditioner",
    "save_raw_seed_initial_artifacts",
    "stage_modes_for",
    "stage_promotes_candidate",
    "target_helicity_seed_terms",
    "write_qi_stage_checkpoint",
]


@dataclass(frozen=True)
class QIOptimizationContext:
    """Explicit staged-QI controls shared by helper routines.

    The example scripts still show the optimization workflow directly. This
    context only replaces the older ``configure(globals())`` helper plumbing so
    source users can reason about which controls are passed into staged helpers.
    """

    alpha: float
    continuation_nfev: int
    inner_max_iter: int
    jit_booz: bool
    max_elongation: float
    max_mirror_ratio: float
    max_mode: int
    max_nfev: int
    method: str
    min_vmec_mode: int
    mirror_surface_index: object
    mirror_weight: float
    opt_qi_resolution: dict
    output_dir: Path
    qi_gate_legacy_max: float
    qi_gate_smooth_max: float
    qi_options: object
    qi_weight: float
    scalar_cost_only_trials: bool | None
    scipy_lsmr_maxiter: int | None
    solver_device: str | None
    stage_modes: tuple
    stage_repeats: int
    surfaces: object
    target_abs_iota_min: float
    target_aspect: float
    trial_ftol: float
    use_ess: bool
    use_mode_continuation: bool


_CONTEXT_FIELDS = {
    "alpha": "ALPHA",
    "continuation_nfev": "CONTINUATION_NFEV",
    "inner_max_iter": "INNER_MAX_ITER",
    "jit_booz": "JIT_BOOZ",
    "max_elongation": "MAX_ELONGATION",
    "max_mirror_ratio": "MAX_MIRROR_RATIO",
    "max_mode": "MAX_MODE",
    "max_nfev": "MAX_NFEV",
    "method": "METHOD",
    "min_vmec_mode": "MIN_VMEC_MODE",
    "mirror_surface_index": "MIRROR_SURFACE_INDEX",
    "mirror_weight": "MIRROR_WEIGHT",
    "opt_qi_resolution": "OPT_QI_RESOLUTION",
    "output_dir": "OUTPUT_DIR",
    "qi_gate_legacy_max": "QI_GATE_LEGACY_MAX",
    "qi_gate_smooth_max": "QI_GATE_SMOOTH_MAX",
    "qi_options": "QI_OPTIONS",
    "qi_weight": "QI_WEIGHT",
    "scalar_cost_only_trials": "SCALAR_COST_ONLY_TRIALS",
    "scipy_lsmr_maxiter": "SCIPY_LSMR_MAXITER",
    "solver_device": "SOLVER_DEVICE",
    "stage_modes": "STAGE_MODES",
    "stage_repeats": "STAGE_REPEATS",
    "surfaces": "SURFACES",
    "target_abs_iota_min": "TARGET_ABS_IOTA_MIN",
    "target_aspect": "TARGET_ASPECT",
    "trial_ftol": "TRIAL_FTOL",
    "use_ess": "USE_ESS",
    "use_mode_continuation": "USE_MODE_CONTINUATION",
}

_DEFAULT_CONTEXT: QIOptimizationContext | None = None


def _float_tuple(value: str) -> tuple[float, ...]:
    text = str(value).strip()
    if not text:
        return ()
    return tuple(float(part.strip()) for part in text.split(",") if part.strip())


def qi_stage_modes(
    *,
    max_mode: int,
    use_mode_continuation: bool,
    continuation_nfev: int,
    repeats: int = 3,
    policy: str = "lower",
) -> list[int]:
    """Return the stage mode sequence for the QI example workflow.

    ``policy="lower"`` uses the same lower-mode continuation semantics as the
    QA/QH/QP examples. ``policy="lower-repeat"`` repeats each lower-mode rung,
    which is useful for far circular seeds because each active spectral shell
    gets cleanup passes before adding more degrees of freedom. ``policy="repeat"``
    preserves the older QI behavior of repeating only the final mode, which can
    still be useful when the input is already in the right basin.
    """

    policy_key = str(policy).strip().lower().replace("_", "-")
    if policy_key in {"lower", "lower-mode", "mode", "qs"}:
        return vj.qs_stage_modes(
            max_mode=max_mode,
            use_mode_continuation=use_mode_continuation,
            continuation_nfev=continuation_nfev,
        )
    if policy_key in {"lower-repeat", "ladder-repeat", "repeat-lower", "rung-repeat"}:
        if not (bool(use_mode_continuation) and int(max_mode) > 1 and int(continuation_nfev) > 0):
            return [int(max_mode)]
        repeats_i = max(1, int(repeats))
        return [int(mode) for mode in range(1, int(max_mode) + 1) for _ in range(repeats_i)]
    if policy_key in {"repeat", "same", "same-mode", "same-mode-repeat"}:
        return vj.repeated_stage_modes(
            max_mode=max_mode,
            use_mode_continuation=use_mode_continuation,
            continuation_nfev=continuation_nfev,
            repeats=repeats,
        )
    raise ValueError("QI stage mode policy must be 'lower', 'lower-repeat', or 'repeat'.")


def qi_engineering_constraint_tuples(
    mirror,
    elongation,
    stage,
    mirror_weight: float,
    elongation_weight: float,
) -> list[tuple]:
    """Return mirror/elongation objective tuples for one QI stage.

    Ordinary stages use the raw upper-bound penalties.  Stages with
    ``use_augmented_lagrangian_constraints=True`` wrap the signed mirror and
    elongation constraints using :class:`vmec_jax.AugmentedLagrangianConstraint`,
    which keeps the driver script compact while exposing the same
    SIMSOPT-style tuple workflow to users.
    """

    stage = {} if stage is None else stage
    mirror_weight = float(mirror_weight)
    elongation_weight = float(elongation_weight)
    if not bool(stage.get("use_augmented_lagrangian_constraints", False)):
        return [
            (mirror.J, 0.0, mirror_weight),
            (elongation.J, 0.0, elongation_weight),
        ]

    softness = float(stage.get("al_constraint_softness", 0.0))

    def _al_tuple(objective, prefix: str, default_weight: float):
        multiplier = float(stage.get(f"al_{prefix}_multiplier", 0.0))
        penalty = float(stage.get(f"al_{prefix}_penalty", 1.0))
        weight = float(stage.get(f"al_{prefix}_weight", default_weight if default_weight > 0.0 else 1.0))
        wrapped = vj.AugmentedLagrangianConstraint(
            objective,
            multiplier=multiplier,
            penalty=penalty,
            softness=softness,
            name=f"al_{prefix}",
        )
        return (wrapped.J, 0.0, weight)

    out = []
    if mirror_weight > 0.0 or float(stage.get("al_mirror_weight", 0.0)) > 0.0:
        out.append(_al_tuple(mirror, "mirror", mirror_weight))
    if elongation_weight > 0.0 or float(stage.get("al_elongation_weight", 0.0)) > 0.0:
        out.append(_al_tuple(elongation, "elongation", elongation_weight))
    return out


def qi_mirror_objective_for_stage(
    stage,
    *,
    qi_options,
    threshold: float,
    surfaces,
    surface_index,
    ntheta: int = 96,
    nphi: int = 96,
):
    """Return the mirror objective requested by one QI optimization stage.

    ``mirror_backend="vmec"`` keeps the fast VMEC-grid mirror penalty used by
    broad exploratory stages. ``mirror_backend="boozer"`` evaluates mirror
    ratio in Boozer coordinates through the shared QI field.
    ``mirror_backend="boozer_scalar"`` evaluates a separate scalar Boozer
    mirror objective, allowing final cleanup to use audit-grade ``mboz/nboz``
    without increasing the whole QI residual resolution.
    """

    stage = {} if stage is None else stage
    backend = str(stage.get("mirror_backend", stage.get("mirror_coordinate", "vmec"))).strip().lower()
    kwargs = dict(
        threshold=float(stage.get("mirror_threshold", threshold)),
        surfaces=surfaces,
        ntheta=int(stage.get("mirror_ntheta", ntheta)),
        nphi=int(stage.get("mirror_nphi", nphi)),
        surface_index=stage.get("mirror_surface_index", surface_index),
        smooth_extrema=float(stage.get("mirror_smooth_extrema", 2.0e-2)),
        smooth_penalty=float(stage.get("mirror_smooth_penalty", 2.0e-2)),
    )
    if backend in {"boozer", "booz", "boozer-qi", "qi"}:
        return vj.MirrorRatio(**kwargs, qi_options=qi_options)
    if backend in {"boozer-scalar", "boozer_scalar", "booz-scalar", "booz_scalar"}:
        return vj.MirrorRatio(
            **kwargs,
            mboz=int(stage.get("mirror_mboz", stage.get("mboz", 18))),
            nboz=int(stage.get("mirror_nboz", stage.get("nboz", 18))),
            jit_booz=bool(stage.get("mirror_jit_booz", True)),
        )
    if backend in {"vmec", "realspace", "real-space"}:
        return vj.VMECMirrorRatio(**kwargs)
    raise ValueError("QI mirror_backend must be 'vmec', 'boozer', or 'boozer_scalar'.")


def apply_qi_example_cli_overrides(namespace: dict, argv: list[str] | None = None) -> argparse.Namespace:
    """Apply optional command-line overrides to a QI example namespace.

    The QI example remains editable by changing top-level variables.  Sweep
    drivers can call the same script with explicit CLI overrides, avoiding the
    older environment-variable control path.
    """

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--input-file", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-mode", type=int)
    parser.add_argument("--min-vmec-mode", type=int)
    parser.add_argument("--max-nfev", type=int)
    parser.add_argument("--continuation-nfev", type=int)
    parser.add_argument(
        "--method",
        choices=(
            "auto",
            "auto_scalar",
            "gauss_newton",
            "scipy",
            "scipy_matrix_free",
            "lbfgs_adjoint",
            "scalar_trust",
        ),
    )
    parser.add_argument("--ftol", type=float)
    parser.add_argument("--gtol", type=float)
    parser.add_argument("--xtol", type=float)
    parser.add_argument("--inner-max-iter", type=int)
    parser.add_argument("--inner-ftol", type=float)
    parser.add_argument("--trial-max-iter", type=int)
    parser.add_argument("--trial-ftol", type=float)
    parser.add_argument("--solver-device", choices=("cpu", "gpu", "none", "default"))
    parser.add_argument("--ess-alpha", type=float)
    parser.add_argument("--use-ess", action=argparse.BooleanOptionalAction)
    parser.add_argument("--use-mode-continuation", action=argparse.BooleanOptionalAction)
    parser.add_argument("--use-simple-seed", action=argparse.BooleanOptionalAction)
    parser.add_argument("--use-target-helicity-seed", action=argparse.BooleanOptionalAction)
    parser.add_argument("--use-reference-family-seed", action=argparse.BooleanOptionalAction)
    parser.add_argument("--reference-input", type=Path)
    parser.add_argument("--reference-lambdas", type=_float_tuple)
    parser.add_argument("--accept-boundary-reference-baseline", action=argparse.BooleanOptionalAction)
    parser.add_argument("--stage-repeats", type=int)
    parser.add_argument("--stage-mode-policy", choices=("lower", "lower-repeat", "repeat"))
    parser.add_argument("--scipy-lsmr-maxiter", type=int)
    parser.add_argument("--scalar-cost-only-trials", action=argparse.BooleanOptionalAction)
    parser.add_argument("--make-plots", action=argparse.BooleanOptionalAction)
    parser.add_argument("--jit-booz", action=argparse.BooleanOptionalAction)
    parser.add_argument("--qi-mboz", type=int)
    parser.add_argument("--qi-nboz", type=int)
    parser.add_argument("--qi-nphi", type=int)
    parser.add_argument("--qi-nalpha", type=int)
    parser.add_argument("--qi-n-bounce", type=int)
    parser.add_argument("--target-aspect", type=float)
    parser.add_argument("--target-abs-iota-min", type=float)
    parser.add_argument("--max-mirror-ratio", type=float)
    parser.add_argument("--max-elongation", type=float)
    parser.add_argument("--mirror-surface-index")
    parser.add_argument("--mirror-weight", type=float)
    parser.add_argument("--elongation-weight", type=float)
    parser.add_argument("--qi-gate-smooth-max", type=float)
    parser.add_argument("--qi-gate-legacy-max", type=float)
    parser.add_argument("--qi-ceiling-max", type=float)
    parser.add_argument("--qi-ceiling-smooth-penalty", type=float)
    parser.add_argument("--audit-qi-mboz", type=int)
    parser.add_argument("--audit-qi-nboz", type=int)
    parser.add_argument("--audit-qi-nphi", type=int)
    parser.add_argument("--audit-qi-nalpha", type=int)
    parser.add_argument("--audit-qi-n-bounce", type=int)
    parser.add_argument("--boundary-reference-json", type=Path)
    parser.add_argument("--mirror-ramp-stages-json", type=Path)
    args, _unknown = parser.parse_known_args(argv)

    def set_if(name: str, value) -> None:
        if value is not None:
            namespace[name] = value

    if args.input_file is not None:
        namespace["INPUT_FILE"] = args.input_file.expanduser()
    if args.output_dir is not None:
        namespace["OUTPUT_DIR"] = args.output_dir.expanduser()
    set_if("MAX_MODE", None if args.max_mode is None else int(args.max_mode))
    namespace["MIN_VMEC_MODE"] = (
        int(args.min_vmec_mode)
        if args.min_vmec_mode is not None
        else max(6, int(namespace["MAX_MODE"]) + 3)
    )
    set_if("MAX_NFEV", None if args.max_nfev is None else int(args.max_nfev))
    set_if("CONTINUATION_NFEV", None if args.continuation_nfev is None else int(args.continuation_nfev))
    set_if("METHOD", args.method)
    set_if("FTOL", None if args.ftol is None else float(args.ftol))
    set_if("GTOL", None if args.gtol is None else float(args.gtol))
    set_if("XTOL", None if args.xtol is None else float(args.xtol))
    set_if("INNER_MAX_ITER", None if args.inner_max_iter is None else int(args.inner_max_iter))
    set_if("INNER_FTOL", None if args.inner_ftol is None else float(args.inner_ftol))
    set_if("TRIAL_MAX_ITER", None if args.trial_max_iter is None else int(args.trial_max_iter))
    set_if("TRIAL_FTOL", None if args.trial_ftol is None else float(args.trial_ftol))
    if args.solver_device is not None:
        namespace["SOLVER_DEVICE"] = None if args.solver_device in {"none", "default"} else str(args.solver_device)
    set_if("ALPHA", None if args.ess_alpha is None else float(args.ess_alpha))
    set_if("USE_ESS", args.use_ess)
    set_if("USE_MODE_CONTINUATION", args.use_mode_continuation)
    set_if("USE_SIMPLE_SEED", args.use_simple_seed)
    set_if("USE_TARGET_HELICITY_SEED", args.use_target_helicity_seed)
    if args.reference_input is not None:
        namespace["REFERENCE_INPUT_FILE"] = args.reference_input.expanduser()
        if args.use_reference_family_seed is None:
            namespace["USE_REFERENCE_FAMILY_SEED"] = True
    set_if("USE_REFERENCE_FAMILY_SEED", args.use_reference_family_seed)
    set_if("REFERENCE_LAMBDAS", args.reference_lambdas)
    set_if("BOUNDARY_REFERENCE_ACCEPT_AS_BASELINE", args.accept_boundary_reference_baseline)
    set_if("STAGE_REPEATS", None if args.stage_repeats is None else int(args.stage_repeats))
    set_if("STAGE_MODE_POLICY", args.stage_mode_policy)
    set_if("SCIPY_LSMR_MAXITER", None if args.scipy_lsmr_maxiter is None else int(args.scipy_lsmr_maxiter))
    set_if("SCALAR_COST_ONLY_TRIALS", args.scalar_cost_only_trials)
    set_if("MAKE_PLOTS", args.make_plots)
    set_if("JIT_BOOZ", args.jit_booz)
    qi_resolution_updates = {
        "mboz": args.qi_mboz,
        "nboz": args.qi_nboz,
        "nphi": args.qi_nphi,
        "nalpha": args.qi_nalpha,
        "n_bounce": args.qi_n_bounce,
    }
    if any(value is not None for value in qi_resolution_updates.values()):
        opt_resolution = dict(namespace.get("OPT_QI_RESOLUTION", {}))
        audit_resolution = dict(namespace.get("AUDIT_QI_RESOLUTION", opt_resolution))
        for key, value in qi_resolution_updates.items():
            if value is not None:
                opt_resolution[key] = int(value)
                audit_resolution[key] = int(value)
        namespace["OPT_QI_RESOLUTION"] = opt_resolution
        namespace["AUDIT_QI_RESOLUTION"] = audit_resolution
    set_if("TARGET_ASPECT", None if args.target_aspect is None else float(args.target_aspect))
    set_if("TARGET_ABS_IOTA_MIN", None if args.target_abs_iota_min is None else float(args.target_abs_iota_min))
    set_if("MAX_MIRROR_RATIO", None if args.max_mirror_ratio is None else float(args.max_mirror_ratio))
    set_if("MAX_ELONGATION", None if args.max_elongation is None else float(args.max_elongation))
    if args.mirror_surface_index is not None:
        text = str(args.mirror_surface_index).strip().lower()
        namespace["MIRROR_SURFACE_INDEX"] = None if text in {"", "none", "null"} else int(text)
    set_if("MIRROR_WEIGHT", None if args.mirror_weight is None else float(args.mirror_weight))
    set_if("ELONGATION_WEIGHT", None if args.elongation_weight is None else float(args.elongation_weight))
    set_if("QI_GATE_SMOOTH_MAX", None if args.qi_gate_smooth_max is None else float(args.qi_gate_smooth_max))
    set_if("QI_GATE_LEGACY_MAX", None if args.qi_gate_legacy_max is None else float(args.qi_gate_legacy_max))
    set_if("QI_CEILING_MAX", None if args.qi_ceiling_max is None else float(args.qi_ceiling_max))
    set_if(
        "QI_CEILING_SMOOTH_PENALTY",
        None if args.qi_ceiling_smooth_penalty is None else float(args.qi_ceiling_smooth_penalty),
    )
    audit_resolution_updates = {
        "mboz": args.audit_qi_mboz,
        "nboz": args.audit_qi_nboz,
        "nphi": args.audit_qi_nphi,
        "nalpha": args.audit_qi_nalpha,
        "n_bounce": args.audit_qi_n_bounce,
    }
    if any(value is not None for value in audit_resolution_updates.values()):
        audit_resolution = dict(namespace.get("AUDIT_QI_RESOLUTION", namespace.get("OPT_QI_RESOLUTION", {})))
        for key, value in audit_resolution_updates.items():
            if value is not None:
                audit_resolution[key] = int(value)
        namespace["AUDIT_QI_RESOLUTION"] = audit_resolution
    if args.boundary_reference_json is not None:
        reference_path = args.boundary_reference_json.expanduser()
        try:
            reference_overrides = json.loads(reference_path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid --boundary-reference-json file: {reference_path}") from exc
        if not isinstance(reference_overrides, dict):
            raise ValueError("--boundary-reference-json must contain a JSON object.")
        if reference_overrides.get("reference_input") is not None:
            reference_overrides["reference_input"] = Path(reference_overrides["reference_input"]).expanduser()
        if reference_overrides.get("lambdas") is not None:
            reference_overrides["lambdas"] = tuple(float(value) for value in reference_overrides["lambdas"])
        namespace["BOUNDARY_REFERENCE_OVERRIDES"] = reference_overrides
    if args.mirror_ramp_stages_json is not None:
        stages_path = args.mirror_ramp_stages_json.expanduser()
        try:
            stages = json.loads(stages_path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid --mirror-ramp-stages-json file: {stages_path}") from exc
        if not isinstance(stages, list) or not all(isinstance(stage, dict) for stage in stages):
            raise ValueError("--mirror-ramp-stages-json must contain a JSON list of stage dictionaries.")
        namespace["MIRROR_RAMP_STAGES"] = tuple(stages)
    namespace["STAGE_MODES"] = qi_stage_modes(
        max_mode=int(namespace["MAX_MODE"]),
        use_mode_continuation=bool(namespace["USE_MODE_CONTINUATION"]),
        continuation_nfev=int(namespace["CONTINUATION_NFEV"]),
        repeats=int(namespace["STAGE_REPEATS"]),
        policy=str(namespace.get("STAGE_MODE_POLICY", "lower")),
    )
    return args


def make_qi_optimization_context(
    context: dict | None = None,
    /,
    *,
    strict: bool = False,
    **overrides,
) -> QIOptimizationContext:
    """Build a typed staged-QI helper context from script constants.

    ``strict=True`` is recommended for standalone examples: every required
    control must be present in ``context`` or ``overrides``.  The default
    ``strict=False`` preserves compatibility with older scripts that installed
    constants with :func:`configure` and relied on module globals.
    """

    values = {}
    if context:
        values.update(context)
    values.update(overrides)

    def get(field: str):
        upper = _CONTEXT_FIELDS[field]
        if field in values:
            return values[field]
        if upper in values:
            return values[upper]
        if not strict and upper in globals():
            return globals()[upper]
        raise KeyError(upper)

    def get_optional(field: str, default=None):
        try:
            return get(field)
        except KeyError:
            return default

    return QIOptimizationContext(
        alpha=float(get("alpha")),
        continuation_nfev=int(get("continuation_nfev")),
        inner_max_iter=int(get("inner_max_iter")),
        jit_booz=bool(get("jit_booz")),
        max_elongation=float(get("max_elongation")),
        max_mirror_ratio=float(get("max_mirror_ratio")),
        max_mode=int(get("max_mode")),
        max_nfev=int(get("max_nfev")),
        method=str(get("method")),
        min_vmec_mode=int(get("min_vmec_mode")),
        mirror_surface_index=get("mirror_surface_index"),
        mirror_weight=float(get("mirror_weight")),
        opt_qi_resolution=dict(get("opt_qi_resolution") or {}),
        output_dir=Path(get("output_dir")),
        qi_gate_legacy_max=float(get("qi_gate_legacy_max")),
        qi_gate_smooth_max=float(get("qi_gate_smooth_max")),
        qi_options=get("qi_options"),
        qi_weight=float(get("qi_weight")),
        scalar_cost_only_trials=get_optional("scalar_cost_only_trials", None),
        scipy_lsmr_maxiter=get_optional("scipy_lsmr_maxiter", None),
        solver_device=get("solver_device"),
        stage_modes=tuple(get("stage_modes")),
        stage_repeats=int(get("stage_repeats")),
        surfaces=get("surfaces"),
        target_abs_iota_min=float(get("target_abs_iota_min")),
        target_aspect=float(get("target_aspect")),
        trial_ftol=float(get("trial_ftol")),
        use_ess=bool(get("use_ess")),
        use_mode_continuation=bool(get("use_mode_continuation")),
    )


def _resolve_context(ctx: QIOptimizationContext | None = None) -> QIOptimizationContext | None:
    return _DEFAULT_CONTEXT if ctx is None else ctx


def _ctx(ctx: QIOptimizationContext | None, field: str):
    resolved = _resolve_context(ctx)
    if resolved is not None:
        return getattr(resolved, field)
    return globals()[_CONTEXT_FIELDS[field]]


def configure(context: dict) -> None:
    """Install script-level constants used by the staged helper routines."""

    global _DEFAULT_CONTEXT
    globals().update(context)
    try:
        _DEFAULT_CONTEXT = make_qi_optimization_context(context)
    except KeyError:
        _DEFAULT_CONTEXT = None


def _diagnostic_float(record, key):
    value = record.get(key)
    return float(value) if value is not None else float("nan")


def diagnostic_float(record, key):
    """Return a scalar diagnostic value, or ``nan`` when it is unavailable."""

    return _diagnostic_float(record, key)


def _finite_or_inf(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("inf")
    return out if np.isfinite(out) else float("inf")


def _finite_or_none(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _parse_float_sequence(value, *, name):
    """Parse a comma/space separated sequence used by subprocess wrappers."""

    if value in (None, ""):
        return None
    pieces = str(value).replace(",", " ").split()
    if not pieces:
        return None
    try:
        return tuple(float(piece) for piece in pieces)
    except ValueError as exc:
        raise ValueError(f"{name} must be a comma- or space-separated float list: {value!r}") from exc


def _resolution_value(resolution, key, default):
    """Return an integer QI diagnostic resolution from a case override mapping."""

    return int(dict(resolution).get(key, default))


TARGET_HELICITY_SEED_AMPLITUDE = 1.0e-5
TARGET_HELICITY_SEED_MODE_TERMS = (
    ("RBC", (1, 0)),
    ("ZBS", (1, 0)),
    ("RBC", (-1, 1)),
    ("ZBS", (-1, 1)),
    ("RBC", (1, 1)),
    ("ZBS", (1, 1)),
)


def target_helicity_seed_terms(*, max_mode, amplitude=TARGET_HELICITY_SEED_AMPLITUDE):
    """Return deterministic low-order perturbations for circular/minimal seeds."""

    if float(amplitude) == 0.0 or int(max_mode) < 1:
        return ()
    return tuple(
        (name, index, float(amplitude))
        for name, index in TARGET_HELICITY_SEED_MODE_TERMS
        if max(abs(int(index[0])), abs(int(index[1]))) <= int(max_mode)
    )


def _normalise_seed_terms(config, *, ctx: QIOptimizationContext | None = None):
    """Return target-helicity seed terms from case config or compact tuples."""

    if not config:
        return ()
    if isinstance(config, dict):
        if not bool(config.get("enabled", True)):
            return ()
        terms = config.get("terms")
        if terms is None:
            terms = target_helicity_seed_terms(
                max_mode=int(config.get("max_mode", _ctx(ctx, "max_mode"))),
                amplitude=float(config.get("amplitude", TARGET_HELICITY_SEED_AMPLITUDE)),
            )
    else:
        terms = config
    normalised = []
    for name, index, value in terms:
        normalised.append((str(name).upper(), (int(index[0]), int(index[1])), float(value)))
    return tuple(normalised)


def run_target_helicity_seed_preconditioner(input_file, output_dir, config, *, ctx: QIOptimizationContext | None = None):
    """Insert deterministic 1e-5 target-helicity modes before local QI solves.

    The source VMEC input is left untouched.  Existing nonzero coefficients are
    preserved by default, so reviewed QI reference inputs are not perturbed while
    circular/minimal seeds get a reproducible non-axisymmetric derivative seed.
    """

    config = dict(config or {}) if isinstance(config, dict) else {"terms": config}
    terms = _normalise_seed_terms(config, ctx=ctx)
    if not terms:
        return Path(input_file)
    source = vj.read_indata(input_file)
    indexed = {name: dict(values) for name, values in source.indexed.items()}
    threshold = float(config.get("only_if_abs_below", 0.0))
    inserted = []
    for name, index, value in terms:
        coeffs = indexed.setdefault(name, {})
        existing = coeffs.get(index)
        try:
            existing_abs = abs(float(existing)) if existing is not None else 0.0
        except (TypeError, ValueError):
            existing_abs = 0.0
        if existing_abs <= threshold:
            coeffs[index] = float(value)
            inserted.append((name, index, float(value)))

    seed_dir = Path(output_dir) / "target_helicity_seed"
    seed_dir.mkdir(parents=True, exist_ok=True)
    input_out = seed_dir / "input.target_helicity_seed"
    seeded = InData(scalars=dict(source.scalars), indexed=indexed, source_path=str(input_file))
    vj.write_indata(input_out, seeded)
    metadata = {
        "enabled": True,
        "source_input": str(input_file),
        "seeded_input": str(input_out),
        "only_if_abs_below": threshold,
        "terms": [
            {"family": name, "n": int(index[0]), "m": int(index[1]), "value": float(value)}
            for name, index, value in terms
        ],
        "inserted": [
            {"family": name, "n": int(index[0]), "m": int(index[1]), "value": float(value)}
            for name, index, value in inserted
        ],
    }
    (seed_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print("\nTarget-helicity seed preconditioner:")
    print(f"  source:   {input_file}")
    print(f"  inserted: {len(inserted)} / {len(terms)} terms")
    print(f"  input:    {input_out}")
    return input_out


def _jsonable(value):
    """Convert NumPy/JAX-like values into JSON-serializable containers."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    try:
        arr = np.asarray(value)
    except Exception:
        return str(value)
    if arr.ndim == 0:
        return _jsonable(arr.item())
    return _jsonable(arr.tolist())


def jsonable(value):
    """Convert NumPy/JAX-like values into JSON-serializable containers."""

    return _jsonable(value)


def _write_json_atomic(path, payload) -> None:
    """Write a JSON artifact via replace so interrupted writes do not corrupt it."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _stage_result_history(stage_result) -> dict:
    """Return the history payload from either workflow results or test doubles."""

    history = getattr(stage_result, "history", None)
    if history is None and isinstance(stage_result, dict):
        history = stage_result.get("_history_dump")
    if history is None:
        final_result = getattr(stage_result, "final_result", {})
        if isinstance(final_result, dict):
            history = final_result.get("_history_dump")
    return dict(history or {})


def _partial_diagnostics_from_history(history: dict, diagnostics: dict) -> dict:
    """Keep diagnostics.json useful before expensive independent QI diagnostics run."""

    out = dict(diagnostics)
    had_diagnostics = bool(out)
    added_history_fields = False
    mapping = {
        "objective_final": "objective_final",
        "qs_final": "qs_final",
        "aspect": "aspect_final",
        "mean_iota": "iota_final",
        "nfev": "nfev",
        "njev": "njev",
        "total_wall_time_s": "total_wall_time_s",
    }
    for out_key, history_key in mapping.items():
        value = history.get(history_key)
        if value is not None and out.get(out_key) is None:
            out[out_key] = value
            added_history_fields = True
    if out and added_history_fields:
        out.setdefault("partial", True)
    if out and not had_diagnostics:
        out["partial"] = True
        out["diagnostics_pending"] = True
    return out


def save_raw_seed_initial_artifacts(input_file, input_out, wout_out, *, ctx: QIOptimizationContext | None = None):
    """Save the unpreconditioned VMEC input deck and its solved WOUT."""

    input_out = Path(input_out)
    wout_out = Path(wout_out)
    input_out.parent.mkdir(parents=True, exist_ok=True)
    wout_out.parent.mkdir(parents=True, exist_ok=True)
    vj.write_indata(input_out, vj.read_indata(input_file))
    run = vj.run_fixed_boundary(input_file, solver_device=_ctx(ctx, "solver_device"), verbose=False)
    vj.write_wout_from_fixed_boundary_run(wout_out, run)
    return run


def basin_prefilter_score(metrics, targets, config):
    """Rank prefilter candidates by QI/iota first, engineering second."""

    smooth = _finite_or_inf(metrics.get("qi_smooth_total"))
    legacy = _finite_or_inf(metrics.get("qi_legacy_total"))
    mirror = _finite_or_inf(metrics.get("qi_mirror_ratio_max"))
    elongation = _finite_or_inf(metrics.get("qi_max_elongation"))
    iota = abs(float(metrics.get("mean_iota") or 0.0))
    aspect = _finite_or_inf(metrics.get("aspect"))
    smooth_score = smooth / max(float(targets.smooth_qi_max), 1.0e-16)
    legacy_score = legacy / max(float(targets.legacy_qi_max), 1.0e-16)
    iota_score = max(0.0, float(targets.abs_iota_min) - iota) / max(float(targets.abs_iota_min), 1.0e-16)
    mirror_score = max(0.0, mirror - float(targets.mirror_ratio_max)) / max(float(targets.mirror_ratio_max), 1.0e-16)
    elongation_score = max(0.0, elongation - float(targets.max_elongation)) / max(float(targets.max_elongation), 1.0e-16)
    aspect_score = abs(aspect - float(targets.target_aspect)) / max(float(targets.target_aspect), 1.0e-16)
    qi_weight = float(config.get("qi_weight", 1.0))
    iota_weight = float(config.get("iota_gap_weight", 3.0))
    mirror_weight = float(config.get("mirror_weight", 0.25))
    elongation_weight = float(config.get("elongation_weight", 0.1))
    aspect_weight = float(config.get("aspect_weight", 0.1))
    return float(
        qi_weight * (smooth_score + legacy_score)
        + iota_weight * iota_score
        + mirror_weight * mirror_score
        + elongation_weight * elongation_score
        + aspect_weight * aspect_score
    )


def make_basin_prefilter_options(config, *, ctx: QIOptimizationContext | None = None):
    qi_options = _ctx(ctx, "qi_options")
    return vj.QIDiagnosticOptions(
        surfaces=_ctx(ctx, "surfaces"),
        mboz=_resolution_value(_ctx(ctx, "opt_qi_resolution"), "mboz", qi_options.mboz),
        nboz=_resolution_value(_ctx(ctx, "opt_qi_resolution"), "nboz", qi_options.nboz),
        nphi=_resolution_value(_ctx(ctx, "opt_qi_resolution"), "nphi", qi_options.nphi),
        nalpha=_resolution_value(_ctx(ctx, "opt_qi_resolution"), "nalpha", qi_options.nalpha),
        n_bounce=_resolution_value(_ctx(ctx, "opt_qi_resolution"), "n_bounce", qi_options.n_bounce),
        include_bounce_endpoints=qi_options.include_bounce_endpoints,
        phimin=float(qi_options.phimin),
        jit_booz=_ctx(ctx, "jit_booz"),
        mirror_threshold=float(config.get("mirror_threshold", _ctx(ctx, "max_mirror_ratio"))),
        mirror_ntheta=int(config.get("mirror_ntheta", 32)),
        mirror_nphi=int(config.get("mirror_nphi", 32)),
        mirror_surface_index=config.get("mirror_surface_index", _ctx(ctx, "mirror_surface_index")),
        elongation_threshold=float(config.get("max_elongation", _ctx(ctx, "max_elongation"))),
        elongation_ntheta=int(config.get("elongation_ntheta", 24)),
        elongation_nphi=int(config.get("elongation_nphi", 8)),
    )


def run_basin_prefilter(input_file, output_dir, config, *, ctx: QIOptimizationContext | None = None):
    """Run a bounded large-step prefilter and return the selected input deck."""

    if not bool(config.get("enabled", False)):
        return Path(input_file)
    SurveyTargets, generate_basin_candidates, rank_candidate_records, write_csv, build_diagnostic_stage = (
        _load_basin_prefilter_tools()
    )
    survey_dir = Path(output_dir) / "basin_prefilter"
    survey_dir.mkdir(parents=True, exist_ok=True)
    stage = build_diagnostic_stage(
        input_path=Path(input_file),
        max_mode=_ctx(ctx, "max_mode"),
        min_vmec_mode=_ctx(ctx, "min_vmec_mode"),
        include=("rc", "zs"),
        fix=("rc00",),
        project_input_boundary_to_max_mode=True,
        inner_max_iter=int(config.get("inner_max_iter", 30)),
        inner_ftol=float(config.get("inner_ftol", 1.0e-8)),
        trial_max_iter=int(config.get("trial_max_iter", 30)),
        trial_ftol=float(config.get("trial_ftol", 1.0e-8)),
        solver_device=_ctx(ctx, "solver_device"),
    )
    names = boundary_param_names(stage.specs)
    x_scale = create_x_scale(stage.specs, alpha=float(config.get("alpha", _ctx(ctx, "alpha"))))
    candidates = generate_basin_candidates(
        names=names,
        x_scale=x_scale,
        radii=tuple(float(radius) for radius in config.get("radii", (0.025, 0.05, 0.1))),
        n_random=int(config.get("n_random", 4)),
        rng_seed=int(config.get("rng_seed", 20260515)),
        axis_count=int(config.get("axis_count", 6)),
        directions=tuple(config.get("directions", ("axes", "rademacher"))),
        include_zero=True,
    )[: max(1, int(config.get("max_candidates", 24)))]
    options = make_basin_prefilter_options(config, ctx=ctx)
    targets = SurveyTargets(
        smooth_qi_max=_ctx(ctx, "qi_gate_smooth_max"),
        legacy_qi_max=_ctx(ctx, "qi_gate_legacy_max"),
        mirror_ratio_max=float(config.get("mirror_threshold", _ctx(ctx, "max_mirror_ratio"))),
        max_elongation=float(config.get("max_elongation", _ctx(ctx, "max_elongation"))),
        abs_iota_min=_ctx(ctx, "target_abs_iota_min"),
        target_aspect=_ctx(ctx, "target_aspect"),
        aspect_tolerance=2.0,
    )
    records = []
    for candidate in candidates:
        record = candidate.as_record(names)
        try:
            params = np.asarray(candidate.params, dtype=float)
            state = stage.optimizer._solve_forward(params, trial=True)
            diagnostics = vj.qi_diagnostics_from_state(
                state=state,
                static=stage.ctx.static,
                indata=stage.ctx.indata,
                signgs=stage.ctx.signgs,
                surfaces=options.surfaces,
                options=options,
                flux_local=stage.ctx.flux,
                prof_local={"pressure": stage.ctx.pressure},
                pressure_local=stage.ctx.pressure,
            )
            metrics = {
                "qi_smooth_total": _finite_or_none(diagnostics.get("qi_smooth_total")),
                "qi_legacy_total": _finite_or_none(diagnostics.get("qi_legacy_total")),
                "qi_mirror_ratio_max": _finite_or_none(diagnostics.get("qi_mirror_ratio_max")),
                "qi_max_elongation": _finite_or_none(diagnostics.get("qi_max_elongation")),
                "mean_iota": _finite_or_none(diagnostics.get("mean_iota")),
                "aspect": _finite_or_none(diagnostics.get("aspect")),
            }
            record["metrics"] = metrics
            record["diagnostics"] = diagnostics
            record["prefilter_score"] = basin_prefilter_score(metrics, targets, config)
            if bool(config.get("save_candidate_inputs", True)):
                candidate_dir = survey_dir / "candidates" / candidate.label.replace(":", "_")
                input_out = candidate_dir / "input.candidate"
                stage.optimizer.save_input(input_out, params)
                record["input_path"] = str(input_out)
        except Exception as exc:  # noqa: BLE001 - prefilter keeps failures ranked last.
            record["metrics"] = {}
            record["diagnostics"] = {}
            record["prefilter_score"] = float("inf")
            record["error"] = f"{type(exc).__name__}: {exc}"
        records.append(record)
    ranked = sorted(
        rank_candidate_records(records, targets=targets),
        key=lambda row: (float(row.get("prefilter_score", float("inf"))), float(row.get("score", float("inf")))),
    )
    for rank, record in enumerate(ranked, start=1):
        record["prefilter_rank"] = rank
    top = ranked[: max(1, int(config.get("top_k", 8)))]
    (survey_dir / "candidates.json").write_text(json.dumps(ranked, indent=2, sort_keys=True) + "\n")
    (survey_dir / "top_candidates.json").write_text(json.dumps(top, indent=2, sort_keys=True) + "\n")
    write_csv(ranked, survey_dir / "candidates.csv")
    selected = top[0]
    selected_input = selected.get("input_path")
    if not selected_input:
        selected_input = str(survey_dir / "input.prefilter_selected")
        stage.optimizer.save_input(selected_input, np.asarray(selected["params"], dtype=float))
    print("\nBasin prefilter selected:")
    print(f"  label:          {selected.get('label')}")
    print(f"  prefilter score:{selected.get('prefilter_score')}")
    print(f"  metrics:        {selected.get('metrics')}")
    print(f"  input:          {selected_input}")
    return Path(selected_input)


def qi_diagnostics_for_result(
    stage_result,
    *,
    mirror_threshold,
    mirror_surface_index,
    smooth_qi_max=None,
    legacy_qi_max=None,
    ctx: QIOptimizationContext | None = None,
):
    qi_options = _ctx(ctx, "qi_options")
    surfaces = _ctx(ctx, "surfaces")
    smooth_qi_max = _ctx(ctx, "qi_gate_smooth_max") if smooth_qi_max is None else float(smooth_qi_max)
    legacy_qi_max = _ctx(ctx, "qi_gate_legacy_max") if legacy_qi_max is None else float(legacy_qi_max)
    opt = stage_result.final_optimizer
    diagnostic_options = vj.QIDiagnosticOptions(
        surfaces=surfaces,
        mboz=qi_options.mboz,
        nboz=qi_options.nboz,
        nphi=qi_options.nphi,
        nalpha=qi_options.nalpha,
        n_bounce=qi_options.n_bounce,
        include_bounce_endpoints=qi_options.include_bounce_endpoints,
        softness=qi_options.softness,
        width_weight=qi_options.width_weight,
        branch_width_weight=qi_options.branch_width_weight,
        branch_width_softness=qi_options.branch_width_softness,
        profile_weight=qi_options.profile_weight,
        shuffle_profile_weight=qi_options.shuffle_profile_weight,
        shuffle_profile_softness=qi_options.shuffle_profile_softness,
        shuffle_profile_nphi_out=qi_options.shuffle_profile_nphi_out,
        weighted_shuffle_profile_weight=qi_options.weighted_shuffle_profile_weight,
        weighted_shuffle_profile_softness=qi_options.weighted_shuffle_profile_softness,
        aligned_profile_weight=qi_options.aligned_profile_weight,
        aligned_profile_softness=qi_options.aligned_profile_softness,
        aligned_profile_trap_level=qi_options.aligned_profile_trap_level,
        aligned_profile_trap_softness=qi_options.aligned_profile_trap_softness,
        phimin=float(qi_options.phimin),
        mirror_threshold=mirror_threshold,
        mirror_surface_index=mirror_surface_index,
        elongation_threshold=_ctx(ctx, "max_elongation"),
    )
    diagnostics = vj.qi_diagnostics_from_state(
        state=stage_result.final_state,
        static=opt.static,
        indata=opt.indata,
        signgs=opt.signgs,
        surfaces=surfaces,
        options=diagnostic_options,
    )
    return annotate_qi_seed_suitability(
        diagnostics,
        targets=QISeedSuitabilityTargets(
            smooth_qi_max=smooth_qi_max,
            legacy_qi_max=legacy_qi_max,
            target_aspect=_ctx(ctx, "target_aspect"),
            abs_iota_min=_ctx(ctx, "target_abs_iota_min"),
            mirror_ratio_max=mirror_threshold,
            max_elongation=_ctx(ctx, "max_elongation"),
        ),
    )


def qi_diagnostics_for_run(
    run,
    *,
    mirror_threshold,
    mirror_surface_index,
    target_aspect,
    abs_iota_min,
    max_elongation,
    resolution=None,
    smooth_qi_max=None,
    legacy_qi_max=None,
    ctx: QIOptimizationContext | None = None,
):
    """Independent QI diagnostics for a raw fixed-boundary VMEC run."""

    qi_options = _ctx(ctx, "qi_options")
    surfaces = _ctx(ctx, "surfaces")
    smooth_qi_max = _ctx(ctx, "qi_gate_smooth_max") if smooth_qi_max is None else float(smooth_qi_max)
    legacy_qi_max = _ctx(ctx, "qi_gate_legacy_max") if legacy_qi_max is None else float(legacy_qi_max)
    diagnostic_options = vj.QIDiagnosticOptions(
        surfaces=surfaces,
        mboz=_resolution_value(resolution or {}, "mboz", qi_options.mboz),
        nboz=_resolution_value(resolution or {}, "nboz", qi_options.nboz),
        nphi=_resolution_value(resolution or {}, "nphi", qi_options.nphi),
        nalpha=_resolution_value(resolution or {}, "nalpha", qi_options.nalpha),
        n_bounce=_resolution_value(resolution or {}, "n_bounce", qi_options.n_bounce),
        include_bounce_endpoints=qi_options.include_bounce_endpoints,
        softness=qi_options.softness,
        width_weight=qi_options.width_weight,
        branch_width_weight=qi_options.branch_width_weight,
        branch_width_softness=qi_options.branch_width_softness,
        profile_weight=qi_options.profile_weight,
        shuffle_profile_weight=qi_options.shuffle_profile_weight,
        shuffle_profile_softness=qi_options.shuffle_profile_softness,
        shuffle_profile_nphi_out=qi_options.shuffle_profile_nphi_out,
        weighted_shuffle_profile_weight=qi_options.weighted_shuffle_profile_weight,
        weighted_shuffle_profile_softness=qi_options.weighted_shuffle_profile_softness,
        aligned_profile_weight=qi_options.aligned_profile_weight,
        aligned_profile_softness=qi_options.aligned_profile_softness,
        aligned_profile_trap_level=qi_options.aligned_profile_trap_level,
        aligned_profile_trap_softness=qi_options.aligned_profile_trap_softness,
        phimin=float(qi_options.phimin),
        mirror_threshold=float(mirror_threshold),
        mirror_surface_index=mirror_surface_index,
        elongation_threshold=float(max_elongation),
    )
    diagnostics = vj.qi_diagnostics_from_state(
        state=run.state,
        static=run.static,
        indata=run.indata,
        signgs=run.signgs,
        surfaces=surfaces,
        options=diagnostic_options,
    )
    return annotate_qi_seed_suitability(
        diagnostics,
        targets=QISeedSuitabilityTargets(
            smooth_qi_max=float(smooth_qi_max),
            legacy_qi_max=float(legacy_qi_max),
            target_aspect=float(target_aspect),
            abs_iota_min=float(abs_iota_min),
            mirror_ratio_max=float(mirror_threshold),
            max_elongation=float(max_elongation),
        ),
    )


def write_qi_stage_checkpoint(
    stage_output_dir,
    *,
    stage_index,
    stage_name,
    stage_modes,
    stage_result,
    diagnostics,
    promotion=None,
    role="stage",
    ctx: QIOptimizationContext | None = None,
):
    """Persist QI stage metrics before root finalization can be reached."""

    stage_output_dir = Path(stage_output_dir)
    stage_output_dir.mkdir(parents=True, exist_ok=True)
    history = _stage_result_history(stage_result)
    diagnostics = _partial_diagnostics_from_history(history, dict(diagnostics))
    promotion = {} if promotion is None else dict(promotion)
    provenance = {
        "stage_output_dir": str(stage_output_dir),
        "initial_input_path": str(stage_output_dir / "input.initial"),
        "final_input_path": str(stage_output_dir / "input.final"),
        "initial_wout_path": str(stage_output_dir / "wout_initial.nc"),
        "final_wout_path": str(stage_output_dir / "wout_final.nc"),
    }
    checkpoint = {
        "schema_version": 1,
        "partial": True,
        "role": str(role),
        "stage": None if stage_index is None else int(stage_index),
        "name": str(stage_name),
        "stage_modes": _jsonable(
            [
                {
                    "mode": int(vj.normalize_boundary_mode_limits(mode).mode),
                    "max_m": vj.normalize_boundary_mode_limits(mode).max_m,
                    "max_n": vj.normalize_boundary_mode_limits(mode).max_n,
                    "label": vj.normalize_boundary_mode_limits(mode).label,
                }
                for mode in stage_modes
            ]
        ),
        "history": _jsonable(history),
        "diagnostics": _jsonable(diagnostics),
        "promotion": _jsonable(promotion),
        "provenance": _jsonable(provenance),
        "history_path": str(stage_output_dir / "history.json"),
        "diagnostics_path": str(stage_output_dir / "diagnostics.json"),
        "input_path": str(stage_output_dir / "input.final"),
        "wout_path": str(stage_output_dir / "wout_final.nc"),
        "initial_input_path": provenance["initial_input_path"],
        "final_input_path": provenance["final_input_path"],
        "initial_wout_path": provenance["initial_wout_path"],
        "final_wout_path": provenance["final_wout_path"],
    }
    history_path = stage_output_dir / "history.json"
    diagnostics_path = stage_output_dir / "diagnostics.json"
    checkpoint_path = stage_output_dir / "qi_stage_checkpoint.json"
    if history:
        _write_json_atomic(history_path, history)
    _write_json_atomic(diagnostics_path, diagnostics)
    _write_json_atomic(checkpoint_path, checkpoint)
    _write_json_atomic(_ctx(ctx, "output_dir") / "stage_checkpoint.json", checkpoint)
    return checkpoint_path


def materialize_qi_stage_inputs(stage_output_dir, stage_result):
    """Write root-level stage input files from an optimization result.

    ``least_squares_solve(..., save_final_outputs=False)`` still writes
    per-mode continuation inputs under nested ``stage_*`` directories.  The QI
    staged policy advances between mirror-ramp stages using the mirror-ramp
    root directory, so materialize ``input.final`` there before it becomes the
    next stage's seed.
    """

    stage_output_dir = Path(stage_output_dir)
    stage_output_dir.mkdir(parents=True, exist_ok=True)

    def _get_attr_or_none(name):
        try:
            return getattr(stage_result, name)
        except Exception:
            return None

    def _params(name, result_key=None):
        value = _get_attr_or_none(name)
        if value is not None:
            return value
        final_result = _get_attr_or_none("final_result")
        if isinstance(final_result, dict) and result_key is not None:
            return final_result.get(result_key)
        return None

    def _save(optimizer, path, params):
        if optimizer is None or params is None or not hasattr(optimizer, "save_input"):
            return False
        try:
            optimizer.save_input(path, params)
            return Path(path).exists()
        except Exception:
            return False

    initial_path = stage_output_dir / "input.initial"
    final_path = stage_output_dir / "input.final"
    _save(_get_attr_or_none("initial_optimizer"), initial_path, _params("initial_params"))
    _save(_get_attr_or_none("final_optimizer"), final_path, _params("final_params", "x"))
    return final_path if final_path.exists() else None


def _boundary_reference_checkpoint_diagnostics(output_dir, active_input_file) -> dict:
    """Return selected boundary-reference metrics for a pre-stage checkpoint."""

    summary_path = Path(output_dir) / "boundary_reference_preconditioner" / "summary.json"
    base = {
        "active_input_path": str(active_input_file),
        "partial": True,
        "source": "stage_pending",
    }
    if not summary_path.exists():
        return {**base, "diagnostics_pending": True}
    try:
        records = json.loads(summary_path.read_text())
    except json.JSONDecodeError:
        return {**base, "diagnostics_pending": True}
    if not isinstance(records, list):
        return {**base, "diagnostics_pending": True}

    active_path = str(Path(active_input_file))
    selected = None
    for record in records:
        if not isinstance(record, dict):
            continue
        if bool(record.get("selected")):
            selected = record
            break
        if str(record.get("input")) == active_path:
            selected = record
    if selected is None:
        return {**base, "diagnostics_pending": True}

    return {
        **base,
        "source": "boundary_reference_preconditioner",
        "boundary_reference_input_path": None if selected.get("input") is None else str(selected.get("input")),
        "boundary_reference_wout_path": None if selected.get("wout") is None else str(selected.get("wout")),
        "lambda": _finite_or_none(selected.get("lambda")),
        "aspect": _finite_or_none(selected.get("aspect")),
        "aspect_relative_error": _finite_or_none(selected.get("aspect_relative_error")),
        "mean_iota": _finite_or_none(selected.get("mean_iota")),
        "qi_raw_total": _finite_or_none(selected.get("smooth_qi")),
        "qi_smooth_total": _finite_or_none(selected.get("smooth_qi")),
        "qi_legacy_total": _finite_or_none(selected.get("legacy_qi")),
        "qi_mirror_ratio_max": _finite_or_none(selected.get("mirror")),
        "qi_max_elongation": _finite_or_none(selected.get("elongation")),
        "qi_seed_gate_passed": bool(selected.get("qi_seed_gate_passed")),
        "qi_engineering_gate_passed": bool(selected.get("qi_engineering_gate_passed")),
        "qi_failure_reasons": list(selected.get("failure_reasons", [])),
    }


def boundary_reference_preconditioner_score(
    diagnostics,
    *,
    mirror_selection_weight=0.01,
    constraint_weight=0.25,
    aspect_selection_weight=25.0,
):
    """Rank reference-family candidates by gates first, then exact metrics."""

    engineering_penalty = 0.0 if bool(diagnostics.get("qi_engineering_gate_passed")) else 100.0
    seed_penalty = 0.0 if bool(diagnostics.get("qi_seed_gate_passed")) else 20.0
    rank_score = _finite_or_inf(diagnostics.get("qi_rank_score"))
    constraint_score = _finite_or_inf(diagnostics.get("qi_constraint_score"))
    mirror = _finite_or_none(diagnostics.get("qi_mirror_ratio_max"))
    aspect_relative_error = _finite_or_none(diagnostics.get("aspect_relative_error"))
    return float(
        engineering_penalty
        + seed_penalty
        + rank_score
        + float(constraint_weight) * constraint_score
        + float(mirror_selection_weight) * (0.0 if mirror is None else mirror)
        + float(aspect_selection_weight) * (0.0 if aspect_relative_error is None else aspect_relative_error)
    )


def boundary_reference_record_is_qi_safe(
    record,
    *,
    max_mirror_ratio,
    abs_iota_min,
    target_aspect=None,
    aspect_relative_tolerance=0.25,
):
    """Return whether a preconditioner summary record satisfies safe gates."""

    mirror_ok = _finite_or_inf(record.get("mirror")) <= float(max_mirror_ratio)
    iota_ok = abs(float(record.get("mean_iota") or 0.0)) >= float(abs_iota_min)
    if target_aspect is None:
        aspect_ok = True
    else:
        aspect = _finite_or_inf(record.get("aspect"))
        aspect_ok = abs(aspect - float(target_aspect)) / max(float(target_aspect), 1.0e-16) <= float(
            aspect_relative_tolerance
        )
    return mirror_ok and iota_ok and aspect_ok


def run_boundary_reference_preconditioner(input_file, output_dir, config, *, ctx: QIOptimizationContext | None = None):
    """Scan same-NFP reference-family boundary jumps and return the selected input."""

    config = dict(config or {})
    if not bool(config.get("enabled", False)):
        return Path(input_file)

    pre_dir = Path(output_dir) / "boundary_reference_preconditioner"
    pre_dir.mkdir(parents=True, exist_ok=True)
    reference_input = Path(config["reference_input"]).expanduser()
    seed = vj.read_indata(input_file)
    reference = vj.read_indata(reference_input)
    keys = tuple(str(key).upper() for key in config.get("keys", ("RBC", "ZBS", "RBS", "ZBC")))
    lambdas = tuple(float(value) for value in config.get("lambdas", (0.99, 0.995, 1.0)))
    max_mode = int(config.get("max_mode", _ctx(ctx, "max_mode")))
    records = []
    print("\nBoundary-reference QI preconditioner:")
    print(f"  reference input: {reference_input}")
    print(f"  lambdas:         {lambdas}")
    for lam in lambdas:
        case_dir = pre_dir / f"lambda_{lam:.3f}".replace(".", "p").replace("-", "m")
        case_dir.mkdir(parents=True, exist_ok=True)
        input_out = case_dir / "input.interpolated"
        wout_out = case_dir / "wout_interpolated.nc"
        try:
            candidate = vj.interpolate_indata_boundary(
                seed,
                reference,
                lam,
                keys=keys,
                max_mode=max_mode,
            )
            candidate = vj.rebuild_for_optimization_resolution(
                candidate,
                max_mode=max_mode,
                min_vmec_mode=_ctx(ctx, "min_vmec_mode"),
            )
            vj.write_indata(input_out, candidate)
            run = vj.run_fixed_boundary(
                input_out,
                max_iter=int(config.get("max_iter", _ctx(ctx, "inner_max_iter"))),
                solver_device=_ctx(ctx, "solver_device"),
                verbose=False,
            )
            vj.write_wout_from_fixed_boundary_run(wout_out, run)
            diagnostics = qi_diagnostics_for_run(
                run,
                mirror_threshold=float(config.get("max_mirror_ratio", _ctx(ctx, "max_mirror_ratio"))),
                mirror_surface_index=config.get("mirror_surface_index", _ctx(ctx, "mirror_surface_index")),
                target_aspect=float(config.get("target_aspect", _ctx(ctx, "target_aspect"))),
                abs_iota_min=float(config.get("abs_iota_min", _ctx(ctx, "target_abs_iota_min"))),
                max_elongation=float(config.get("max_elongation", _ctx(ctx, "max_elongation"))),
                resolution=config.get("diagnostic_qi_resolution"),
                smooth_qi_max=float(config.get("smooth_qi_max", _ctx(ctx, "qi_gate_smooth_max"))),
                legacy_qi_max=float(config.get("legacy_qi_max", _ctx(ctx, "qi_gate_legacy_max"))),
                ctx=ctx,
            )
            score = boundary_reference_preconditioner_score(
                diagnostics,
                mirror_selection_weight=float(config.get("mirror_selection_weight", 0.01)),
                constraint_weight=float(config.get("constraint_selection_weight", 0.25)),
                aspect_selection_weight=float(config.get("aspect_selection_weight", 25.0)),
            )
            record = {
                "lambda": lam,
                "input": str(input_out),
                "wout": str(wout_out),
                "score": score,
                "selected": False,
                "smooth_qi": _finite_or_none(diagnostics.get("qi_smooth_total")),
                "legacy_qi": _finite_or_none(diagnostics.get("qi_legacy_total")),
                "mirror": _finite_or_none(diagnostics.get("qi_mirror_ratio_max")),
                "elongation": _finite_or_none(diagnostics.get("qi_max_elongation")),
                "mean_iota": _finite_or_none(diagnostics.get("mean_iota")),
                "aspect": _finite_or_none(diagnostics.get("aspect")),
                "aspect_relative_error": _finite_or_none(diagnostics.get("aspect_relative_error")),
                "qi_seed_gate_passed": bool(diagnostics.get("qi_seed_gate_passed")),
                "qi_engineering_gate_passed": bool(diagnostics.get("qi_engineering_gate_passed")),
                "failure_reasons": list(diagnostics.get("qi_failure_reasons", [])),
            }
            (case_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n")
        except Exception as exc:  # noqa: BLE001 - keep the candidate scan moving.
            record = {
                "lambda": lam,
                "input": str(input_out),
                "wout": str(wout_out),
                "score": float("inf"),
                "selected": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        records.append(record)
        print(
            f"  lambda={lam:.3f}: score={record['score']} "
            f"QI={record.get('legacy_qi')} mirror={record.get('mirror')} "
            f"iota={record.get('mean_iota')} aspect={record.get('aspect')}"
        )

    successful = [record for record in records if np.isfinite(float(record.get("score", float("inf"))))]
    if not successful:
        (pre_dir / "summary.json").write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
        raise RuntimeError("Boundary-reference preconditioner found no successful candidates.")

    candidate_pool = [record for record in successful if bool(record.get("qi_engineering_gate_passed"))] or successful
    target_aspect = float(config.get("target_aspect", _ctx(ctx, "target_aspect")))
    aspect_relative_tolerance = float(config.get("aspect_relative_tolerance", 0.37))
    if bool(config.get("prefer_aspect_candidates", True)):
        aspect_pool = [
            record
            for record in candidate_pool
            if _finite_or_inf(record.get("aspect_relative_error"))
            <= aspect_relative_tolerance
        ]
        if aspect_pool:
            candidate_pool = aspect_pool
    if bool(config.get("prefer_qi_safe_candidates", True)):
        max_mirror_ratio = float(config.get("max_mirror_ratio", _ctx(ctx, "max_mirror_ratio")))
        abs_iota_min = float(config.get("abs_iota_min", _ctx(ctx, "target_abs_iota_min")))
        safe_pool = [
            record
            for record in candidate_pool
            if boundary_reference_record_is_qi_safe(
                record,
                max_mirror_ratio=max_mirror_ratio,
                abs_iota_min=abs_iota_min,
                target_aspect=target_aspect,
                aspect_relative_tolerance=aspect_relative_tolerance,
            )
        ]
        if safe_pool:
            candidate_pool = safe_pool
    if bool(config.get("prefer_non_endpoint", False)):
        non_endpoint = [record for record in candidate_pool if abs(float(record["lambda"]) - 1.0) > 1.0e-12]
        if non_endpoint:
            candidate_pool = non_endpoint
    if bool(config.get("prefer_lowest_qi_candidate", False)):
        selected = min(
            candidate_pool,
            key=lambda record: (
                _finite_or_inf(record.get("smooth_qi")),
                _finite_or_inf(record.get("legacy_qi")),
                float(record["score"]),
            ),
        )
    else:
        selected = min(candidate_pool, key=lambda record: float(record["score"]))
    selected["selected"] = True
    (pre_dir / "summary.json").write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
    print("  selected:       ", selected["input"])
    print(
        "  selected metrics:"
        f" QI={selected.get('legacy_qi')}, mirror={selected.get('mirror')}, "
        f"iota={selected.get('mean_iota')}, aspect={selected.get('aspect')}"
    )
    return Path(selected["input"])


def stage_modes_for(stage, *, ctx: QIOptimizationContext | None = None):
    if "stage_mode_limits" in stage:
        return [vj.normalize_boundary_mode_limits(mode) for mode in stage["stage_mode_limits"]]
    if "stage_modes" in stage:
        return [int(mode) for mode in stage["stage_modes"]]
    return qi_stage_modes(
        max_mode=_ctx(ctx, "max_mode"),
        use_mode_continuation=_ctx(ctx, "use_mode_continuation"),
        continuation_nfev=_ctx(ctx, "continuation_nfev"),
        repeats=int(stage.get("stage_repeats", _ctx(ctx, "stage_repeats"))),
        policy=str(stage.get("stage_mode_policy", "lower")),
    )


def promotion_score(record):
    """Lower score means a better exact-diagnostic QI candidate."""

    seed_penalty = 0.0 if bool(record.get("qi_seed_gate_passed")) else 100.0
    engineering_penalty = 0.0 if bool(record.get("qi_engineering_gate_passed")) else 10.0
    aspect_relative_error = _finite_or_none(record.get("aspect_relative_error"))
    return (
        seed_penalty
        + engineering_penalty
        + _finite_or_inf(record.get("qi_rank_score"))
        + 0.25 * _finite_or_inf(record.get("qi_constraint_score"))
        + 25.0 * (0.0 if aspect_relative_error is None else aspect_relative_error)
    )


def engineering_promotion_score(record):
    """Rank already-gated candidates by QI first and mirror second."""

    seed_penalty = 0.0 if bool(record.get("qi_seed_gate_passed")) else 1000.0
    engineering_penalty = 0.0 if bool(record.get("qi_engineering_gate_passed")) else 100.0
    aspect_relative_error = _finite_or_none(record.get("aspect_relative_error"))
    return (
        seed_penalty
        + engineering_penalty
        + _finite_or_inf(record.get("qi_rank_score"))
        + 0.25 * _finite_or_inf(record.get("qi_constraint_score"))
        + 2.0 * _finite_or_inf(record.get("qi_mirror_ratio_max"))
        + 25.0 * (0.0 if aspect_relative_error is None else aspect_relative_error)
    )


def stage_promotes_candidate(
    stage,
    promotion,
    reference_diagnostics,
    *,
    ctx: QIOptimizationContext | None = None,
):
    """Apply the script's staged promotion rule to exact diagnostics."""

    reasons = list(promotion.get("qi_cleanup_rejection_reasons", []))
    if bool(stage.get("accept_if_iota_improves", False)) and reference_diagnostics is not None:
        candidate_iota = abs(_finite_or_inf(promotion.get("mean_iota")))
        reference_iota = abs(_finite_or_inf(reference_diagnostics.get("mean_iota")))
        iota_gain = candidate_iota - reference_iota
        qi_relax = float(stage.get("qi_relax_for_iota", 2.0))
        smooth_limit = qi_relax * max(
            _ctx(ctx, "qi_gate_smooth_max"),
            _finite_or_inf(reference_diagnostics.get("qi_smooth_total")),
        )
        legacy_limit = qi_relax * max(
            _ctx(ctx, "qi_gate_legacy_max"),
            _finite_or_inf(reference_diagnostics.get("qi_legacy_total")),
        )
        if (
            iota_gain >= float(stage.get("iota_improvement_min", 0.0))
            and _finite_or_inf(promotion.get("qi_smooth_total")) <= smooth_limit
            and _finite_or_inf(promotion.get("qi_legacy_total")) <= legacy_limit
        ):
            out = dict(promotion)
            out["qi_cleanup_promoted"] = True
            out["qi_cleanup_rejection_reasons"] = []
            out["qi_iota_promotion_reason"] = (
                f"iota increased by {iota_gain:.6g} while QI stayed within "
                f"{qi_relax:.3g}x relaxed smooth/legacy limits"
            )
            return out
        reasons.append(
            "iota ramp did not satisfy relaxed QI promotion: "
            f"gain={iota_gain:.6g}, smooth_limit={smooth_limit:.6g}, legacy_limit={legacy_limit:.6g}"
        )
    if bool(stage.get("accept_if_rank_improves", False)) and reference_diagnostics is not None:
        candidate_score = promotion_score(promotion)
        reference_score = promotion_score(reference_diagnostics)
        tolerance = float(stage.get("rank_score_relax", 1.0e-12))
        if candidate_score >= reference_score - tolerance:
            reasons.append(
                "rank score did not improve: "
                f"candidate={candidate_score:.6g}, reference={reference_score:.6g}"
            )
    elif bool(stage.get("accept_if_rank_improves", False)):
        # The first staged far-seed result is allowed to become the baseline.
        pass
    if bool(stage.get("accept_if_engineering_score_improves", False)) and reference_diagnostics is not None:
        candidate_score = engineering_promotion_score(promotion)
        reference_score = engineering_promotion_score(reference_diagnostics)
        candidate_mirror = _finite_or_inf(promotion.get("qi_mirror_ratio_max"))
        reference_mirror = _finite_or_inf(reference_diagnostics.get("qi_mirror_ratio_max"))
        mirror_gain = reference_mirror - candidate_mirror
        if candidate_score >= reference_score - float(stage.get("engineering_score_relax", 1.0e-12)):
            reasons.append(
                "engineering score did not improve: "
                f"candidate={candidate_score:.6g}, reference={reference_score:.6g}"
            )
        if mirror_gain < float(stage.get("mirror_improvement_min", 0.0)):
            reasons.append(
                "mirror ratio did not improve enough: "
                f"gain={mirror_gain:.6g}, required={float(stage.get('mirror_improvement_min', 0.0)):.6g}"
            )
    elif bool(stage.get("accept_if_engineering_score_improves", False)):
        pass
    if reasons:
        out = dict(promotion)
        out["qi_cleanup_promoted"] = False
        out["qi_cleanup_rejection_reasons"] = reasons
        return out
    return promotion


def run_qi_stage_policy(
    active_input_file,
    output_dir,
    *,
    solve_qi_stage,
    make_qi_problem,
    boundary_reference_preconditioner,
    mirror_ramp_stages,
    ctx: QIOptimizationContext | None = None,
):
    """Run the guarded staged QI policy and return ``(result, promotion_log)``.

    The public script still defines the objectives and the solve function. This
    helper only handles repeated staged promotion, exact diagnostics, and
    checkpoint bookkeeping.
    """

    promotion_log = []
    if not mirror_ramp_stages:
        write_qi_stage_checkpoint(
            output_dir,
            stage_index=1,
            stage_name="qi_optimization",
            stage_modes=_ctx(ctx, "stage_modes"),
            stage_result=None,
            diagnostics=_boundary_reference_checkpoint_diagnostics(output_dir, active_input_file),
            promotion={"stage_pending": True},
            role="stage_pending",
            ctx=ctx,
        )
        result = solve_qi_stage(
            active_input_file,
            output_dir,
            make_qi_problem(),
            max_nfev=_ctx(ctx, "max_nfev"),
            label=f"QI optimization (max_mode={_ctx(ctx, 'max_mode')}, {'ESS' if _ctx(ctx, 'use_ess') else 'no ESS'})",
            save_final_outputs=False,
            scipy_lsmr_maxiter=_ctx(ctx, "scipy_lsmr_maxiter"),
            scalar_cost_only_trials=_ctx(ctx, "scalar_cost_only_trials"),
        )
        materialize_qi_stage_inputs(output_dir, result)
        write_qi_stage_checkpoint(
            output_dir,
            stage_index=1,
            stage_name="qi_optimization",
            stage_modes=_ctx(ctx, "stage_modes"),
            stage_result=result,
            diagnostics={},
            promotion={"diagnostics_pending": True},
            role="stage_pre_diagnostics",
            ctx=ctx,
        )
        return result, promotion_log

    accepted_result = None
    accepted_seed_diagnostics = None
    best_result = None
    best_diagnostics = None
    if bool(boundary_reference_preconditioner.get("enabled", False)) and bool(
        boundary_reference_preconditioner.get("accept_as_baseline", False)
    ):
        baseline_output_dir = Path(output_dir) / "boundary_reference_baseline"
        print("\nRecording boundary-reference candidate as accepted baseline ...")
        write_qi_stage_checkpoint(
            baseline_output_dir,
            stage_index=0,
            stage_name="boundary_reference_baseline",
            stage_modes=(_ctx(ctx, "max_mode"),),
            stage_result=None,
            diagnostics=_boundary_reference_checkpoint_diagnostics(output_dir, active_input_file),
            promotion={"stage_pending": True, "baseline": True},
            role="boundary_reference_baseline_pending",
            ctx=ctx,
        )
        accepted_result = solve_qi_stage(
            active_input_file,
            baseline_output_dir,
            make_qi_problem({"qi_weight": _ctx(ctx, "qi_weight"), "qi_ceiling_weight": 0.0}),
            max_nfev=1,
            label=f"QI boundary-reference baseline (max_mode={_ctx(ctx, 'max_mode')})",
            stage_modes=(_ctx(ctx, "max_mode"),),
            method="scipy_matrix_free",
            use_mode_continuation=False,
            scipy_lsmr_maxiter=_ctx(ctx, "scipy_lsmr_maxiter"),
            scalar_cost_only_trials=_ctx(ctx, "scalar_cost_only_trials"),
        )
        materialize_qi_stage_inputs(baseline_output_dir, accepted_result)
        write_qi_stage_checkpoint(
            baseline_output_dir,
            stage_index=0,
            stage_name="boundary_reference_baseline",
            stage_modes=(_ctx(ctx, "max_mode"),),
            stage_result=accepted_result,
            diagnostics={},
            promotion={"qi_cleanup_promoted": True, "baseline": True, "diagnostics_pending": True},
            role="boundary_reference_baseline_pre_diagnostics",
            ctx=ctx,
        )
        accepted_seed_diagnostics = qi_diagnostics_for_result(
            accepted_result,
            mirror_threshold=_ctx(ctx, "max_mirror_ratio"),
            mirror_surface_index=_ctx(ctx, "mirror_surface_index"),
            ctx=ctx,
        )
        write_qi_stage_checkpoint(
            baseline_output_dir,
            stage_index=0,
            stage_name="boundary_reference_baseline",
            stage_modes=(_ctx(ctx, "max_mode"),),
            stage_result=accepted_result,
            diagnostics=accepted_seed_diagnostics,
            promotion={"qi_cleanup_promoted": True, "baseline": True},
            role="boundary_reference_baseline",
            ctx=ctx,
        )
        best_result = accepted_result
        best_diagnostics = accepted_seed_diagnostics

    for stage_index, stage in enumerate(mirror_ramp_stages, start=1):
        stage_name = stage["name"]
        stage_output_dir = Path(output_dir) / f"mirror_ramp_{stage_index:02d}_{stage_name}"
        stage_modes_i = stage_modes_for(stage, ctx=ctx)
        write_qi_stage_checkpoint(
            stage_output_dir,
            stage_index=stage_index,
            stage_name=stage_name,
            stage_modes=stage_modes_i,
            stage_result=None,
            diagnostics=_boundary_reference_checkpoint_diagnostics(output_dir, active_input_file),
            promotion={"stage_pending": True},
            role="mirror_ramp_pending",
            ctx=ctx,
        )
        stage_result = solve_qi_stage(
            active_input_file,
            stage_output_dir,
            make_qi_problem(stage),
            max_nfev=int(stage.get("max_nfev", _ctx(ctx, "max_nfev"))),
            label=f"QI {stage_name} (max_mode={_ctx(ctx, 'max_mode')}, {'ESS' if _ctx(ctx, 'use_ess') else 'no ESS'})",
            stage_modes=stage_modes_i,
            method=str(stage.get("method", _ctx(ctx, "method"))),
            use_mode_continuation=bool(stage.get("use_mode_continuation", _ctx(ctx, "use_mode_continuation"))),
            scalar_step_bound=stage.get("scalar_step_bound"),
            lbfgs_step_bound=stage.get("lbfgs_step_bound"),
            scipy_lsmr_maxiter=stage.get("scipy_lsmr_maxiter", _ctx(ctx, "scipy_lsmr_maxiter")),
            scalar_cost_only_trials=stage.get("scalar_cost_only_trials", _ctx(ctx, "scalar_cost_only_trials")),
        )
        stage_final_input = materialize_qi_stage_inputs(stage_output_dir, stage_result)
        write_qi_stage_checkpoint(
            stage_output_dir,
            stage_index=stage_index,
            stage_name=stage_name,
            stage_modes=stage_modes_i,
            stage_result=stage_result,
            diagnostics={},
            promotion={"diagnostics_pending": True},
            role="mirror_ramp_pre_diagnostics",
            ctx=ctx,
        )
        stage_mirror_threshold = float(stage.get("mirror_threshold", _ctx(ctx, "max_mirror_ratio")))
        stage_promotion_mirror_threshold = float(stage.get("promotion_mirror_threshold", stage_mirror_threshold))
        stage_mirror_surface_index = stage.get("mirror_surface_index", _ctx(ctx, "mirror_surface_index"))
        stage_smooth_qi_max = float(stage.get("smooth_qi_max", _ctx(ctx, "qi_gate_smooth_max")))
        stage_legacy_qi_max = float(stage.get("legacy_qi_max", _ctx(ctx, "qi_gate_legacy_max")))
        reference_diagnostics = None if accepted_result is None else qi_diagnostics_for_result(
            accepted_result,
            mirror_threshold=stage_promotion_mirror_threshold,
            mirror_surface_index=stage_mirror_surface_index,
            smooth_qi_max=stage_smooth_qi_max,
            legacy_qi_max=stage_legacy_qi_max,
            ctx=ctx,
        )
        stage_diagnostics = qi_diagnostics_for_result(
            stage_result,
            mirror_threshold=stage_promotion_mirror_threshold,
            mirror_surface_index=stage_mirror_surface_index,
            smooth_qi_max=stage_smooth_qi_max,
            legacy_qi_max=stage_legacy_qi_max,
            ctx=ctx,
        )
        promotion = vj.qi_cleanup_candidate_promotable(
            stage_diagnostics,
            reference=reference_diagnostics,
            targets=QISeedSuitabilityTargets(
                smooth_qi_max=stage_smooth_qi_max,
                legacy_qi_max=stage_legacy_qi_max,
                target_aspect=_ctx(ctx, "target_aspect"),
                abs_iota_min=_ctx(ctx, "target_abs_iota_min"),
                mirror_ratio_max=stage_promotion_mirror_threshold,
                max_elongation=_ctx(ctx, "max_elongation"),
            ),
            require_seed_gate=bool(stage.get("require_seed_gate", True)),
            require_mirror_improvement=bool(
                stage.get("require_mirror_improvement", accepted_seed_diagnostics is not None)
                and float(stage.get("mirror_weight", _ctx(ctx, "mirror_weight"))) > 0.0
            ),
            require_engineering_gate=bool(stage.get("require_engineering_gate", False)),
            mirror_improvement_min=float(stage.get("mirror_improvement_min", 0.0)),
        )
        promotion = stage_promotes_candidate(stage, promotion, reference_diagnostics, ctx=ctx)
        write_qi_stage_checkpoint(
            stage_output_dir,
            stage_index=stage_index,
            stage_name=stage_name,
            stage_modes=stage_modes_i,
            stage_result=stage_result,
            diagnostics=stage_diagnostics,
            promotion=promotion,
            role="mirror_ramp",
            ctx=ctx,
        )
        promotion_log.append(
            {
                "stage": stage_index,
                "name": stage_name,
                "output_dir": str(stage_output_dir),
                "stage_modes": [_jsonable(vj.normalize_boundary_mode_limits(mode).__dict__) for mode in stage_modes_i],
                "method": str(stage.get("method", _ctx(ctx, "method"))),
                "promoted": bool(promotion["qi_cleanup_promoted"]),
                "smooth_qi": promotion.get("qi_smooth_total"),
                "legacy_qi": promotion.get("qi_legacy_total"),
                "mirror": promotion.get("qi_mirror_ratio_max"),
                "elongation": promotion.get("qi_max_elongation"),
                "mean_iota": promotion.get("mean_iota"),
                "rank_score": promotion.get("qi_rank_score"),
                "constraint_score": promotion.get("qi_constraint_score"),
                "iota_promotion_reason": promotion.get("qi_iota_promotion_reason"),
                "rejection_reasons": promotion.get("qi_cleanup_rejection_reasons", []),
            }
        )
        print(f"\nMirror-ramp stage {stage_index}: {stage_name}")
        print(f"  smooth QI:    {promotion.get('qi_smooth_total')}")
        print(f"  legacy QI:    {promotion.get('qi_legacy_total')}")
        print(f"  mirror ratio: {promotion.get('qi_mirror_ratio_max')}")
        print(f"  elongation:   {promotion.get('qi_max_elongation')}")
        print(f"  mean iota:    {promotion.get('mean_iota')}")
        print(f"  promoted:     {promotion['qi_cleanup_promoted']}")
        for reason in promotion.get("qi_cleanup_rejection_reasons", []):
            print(f"    - {reason}")

        if best_diagnostics is None or promotion_score(stage_diagnostics) < promotion_score(best_diagnostics):
            best_result = stage_result
            best_diagnostics = stage_diagnostics
        if promotion["qi_cleanup_promoted"]:
            accepted_result = stage_result
            accepted_seed_diagnostics = stage_diagnostics
            active_input_file = stage_final_input or stage_output_dir / "input.final"
        elif accepted_result is None:
            print(
                f"Initial QI staged policy {stage_name!r} failed the promotion gate; "
                "continuing with the best exact-diagnostic candidate recorded so far."
            )
            active_input_file = stage_final_input or stage_output_dir / "input.final"
    return (accepted_result if accepted_result is not None else best_result), promotion_log
