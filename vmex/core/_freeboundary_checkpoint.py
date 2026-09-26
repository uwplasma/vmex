"""Lossless accepted-root storage for the free-boundary problem API."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import NamedTuple

import numpy as np

FIELDS = ('R_cos', 'R_sin', 'Z_cos', 'Z_sin', 'L_cos', 'L_sin')
SCHEMA = 'vmex.freeboundary-problem/v1'


def _json(value):
    return json.dumps(value, sort_keys=True, allow_nan=False,
                      default=lambda x: np.asarray(x).tolist())


def identity(inp, chart, constraints, options, context):
    """Bind numerical inputs and caller-supplied objective/optimizer definitions."""
    options = {k: v for k, v in options.items() if k != 'device'}
    return _json(dict(input=asdict(inp), coefficients=chart.coefficients,
                      currents=chart.currents, current_dofs=chart.current_dofs,
                      scales=chart.scales, max_coil_mode=chart.mode,
                      nfp=chart.nfp, stellsym=chart.stellsym, n_segments=chart.n_segments,
                      constraints=[dict(target=c.target, scale=c.scale, rtol=c.rtol, atol=c.atol)
                                   for c in constraints], solver=options, context=context))


def read(path, expected_sha256, expected_identity):
    """Authenticate and validate a checkpoint before any solver is constructed."""
    path = Path(path)
    if not expected_sha256 or hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError('checkpoint SHA256 mismatch')
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name].copy() for name in archive.files}
    if str(data['schema_version']) != SCHEMA:
        raise ValueError('unsupported checkpoint schema; convert historical checkpoints explicitly')
    if str(data['identity']) != expected_identity:
        raise ValueError('checkpoint input, objective, optimizer or solver settings differ')
    step = data['accepted_step']
    if step.shape != () or step.dtype.kind not in 'iu' or int(step) < 0:
        raise ValueError('invalid accepted step')
    for name in ('parameters', 'rcon0', 'zcon0', 'loss_scale', *FIELDS,
                 *('mask_' + f for f in FIELDS)):
        value = data[name]
        if value.dtype != np.float64 or not np.all(np.isfinite(value)):
            raise ValueError('invalid float64 checkpoint field: ' + name)
    if data['parameters'].ndim != 1 or data['loss_scale'].shape != () or data['loss_scale'] <= 0:
        raise ValueError('invalid checkpoint parameters or normalization')
    for name in FIELDS:
        if data[name].ndim != 2 or data[name].shape != data[FIELDS[0]].shape:
            raise ValueError('invalid state shape: ' + name)
        mask = data['mask_' + name]
        if mask.shape != data[name].shape or not np.all(np.isin(mask, (0., 1.))):
            raise ValueError('invalid checkpoint mask: ' + name)
    if data['rcon0'].ndim != 3 or data['zcon0'].shape != data['rcon0'].shape:
        raise ValueError('invalid constraint baselines')
    reference = json.loads(str(data['initial_gradient_norm']))
    if reference is not None and (not np.isfinite(reference) or reference < 0):
        raise ValueError('invalid initial gradient reference')
    data['gradient_reference'] = reference
    return data


def verify(accepted, data):
    """Certification must preserve every saved numerical array exactly."""
    pairs = [(accepted.parameters, data['parameters']),
             (accepted.rcon0, data['rcon0']), (accepted.zcon0, data['zcon0'])]
    pairs += [(getattr(accepted.state, f), data[f]) for f in FIELDS]
    pairs += [(getattr(accepted.dof_mask, f), data['mask_' + f]) for f in FIELDS]
    if any(not np.array_equal(np.asarray(value), saved) for value, saved in pairs):
        raise ValueError('checkpoint state, masks, parameters or baselines changed during certification')


def write(problem, path):
    """Write only the accepted root; existing checkpoint files are preserved."""
    if problem.checkpoint_identity is None:
        raise ValueError('provide checkpoint_identity with objective and optimizer settings')
    accepted = problem.accepted
    data = dict(schema_version=np.asarray(SCHEMA), identity=np.asarray(problem.checkpoint_identity),
                accepted_step=np.asarray(problem.accepted_step),
                initial_gradient_norm=np.asarray(_json(problem.initial_gradient_norm)),
                parameters=np.asarray(accepted.parameters), loss_scale=np.asarray(problem.loss_scale),
                rcon0=np.asarray(accepted.rcon0), zcon0=np.asarray(accepted.zcon0))
    data.update({f: np.asarray(getattr(accepted.state, f)) for f in FIELDS})
    data.update({'mask_' + f: np.asarray(getattr(accepted.dof_mask, f)) for f in FIELDS})
    path = Path(path).resolve()
    with path.open('xb') as stream:
        np.savez_compressed(stream, **data)
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                accepted_step=problem.accepted_step)


class OptimizationQualification(NamedTuple):
    """A passing numerical report and its authenticated, colocated artifacts.

    Numerical tests remain in the verification program. This class only binds
    settings/code/runtime and authenticates the report for production reuse.
    """

    report: dict
    artifacts: dict

    @staticmethod
    def signature(*, parameters, input_path, wout_path=None, sources=(), functions=(), **configuration):
        """Fingerprint the numerical case, independently of its run/reporting code.

        ``functions`` explicitly identifies the problem builder and optimizer
        definitions when they share a file with logging or plotting. Their
        executable syntax is hashed; comments and docstrings are ignored.
        Callers must include every numerical helper/module in ``functions`` or
        ``sources`` and every numerical global in ``parameters``. Whole-file
        hashing remains the default for ``sources``; the core is always bound.
        """
        import ast
        import inspect
        import textwrap
        from importlib.metadata import PackageNotFoundError, version
        import jax

        def digest(path):
            return hashlib.sha256(Path(path).read_bytes()).hexdigest()

        class ExecutableSyntax(ast.NodeTransformer):
            def visit_FunctionDef(self, node):
                self.generic_visit(node)
                if (node.body and isinstance(node.body[0], ast.Expr)
                        and isinstance(node.body[0].value, ast.Constant)
                        and isinstance(node.body[0].value.value, str)):
                    node.body.pop(0)
                return node

        def function_digest(function):
            tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
            syntax = ast.dump(ExecutableSyntax().visit(tree), include_attributes=False)
            return hashlib.sha256(syntax.encode()).hexdigest()

        def dependency_version(name):
            try:
                return version(name)
            except PackageNotFoundError:
                if name not in ("booz_xform_jax", "virtual-casing-jax"):
                    raise
                # Vacuum QA can run without these optional analysis packages.
                # Record their absence explicitly so contracts still differ
                # if one is installed before production or checkpoint reuse.
                return None

        source_paths = [Path(p) for p in sources]
        if len({p.name for p in source_paths}) != len(source_paths):
            raise ValueError("qualification source filenames must be unique")
        functions = tuple(functions)
        if len({f.__name__ for f in functions}) != len(functions):
            raise ValueError("qualification function names must be unique")
        result = dict(parameters=parameters, input_sha256=digest(input_path),
            wout_sha256=None if wout_path is None else digest(wout_path),
            hardware=jax.devices()[0].device_kind, **configuration,
            dependencies={name: dependency_version(name) for name in
                ("jax", "jaxlib", "numpy", "scipy", "essos", "solvax", "booz_xform_jax", "virtual-casing-jax")},
            source_sha256={p.name: digest(p) for p in source_paths},
            core_sha256={p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))})
        if functions:
            result["numerical_functions_sha256"] = {f.__name__: function_digest(f) for f in functions}
        return json.loads(_json(result))

    @classmethod
    def read(cls, path, *, contract, required_checks=("finite_difference", "matrix_free")):
        """Require passing checks, an identical contract, and unchanged artifacts."""
        path = Path(path).resolve()
        report = json.loads(path.read_text())
        if report.get("schema") != "vmex.single-stage-qualification/v1" or report.get("passed") is not True:
            raise ValueError("qualification report has not passed")
        if any(report.get("checks", {}).get(name, {}).get("passed") is not True for name in required_checks):
            raise ValueError("qualification checks have not passed")
        if report.get("contract") != contract:
            raise ValueError("qualification differs from current code, input, physics, resolution or runtime")
        assets = {}
        for name in ("coils", "checkpoint"):
            item = report["artifacts"][name]
            filename = item["file"]
            if not filename or Path(filename).name != filename or filename in (".", ".."):
                raise ValueError("qualification artifacts must be files beside the report")
            asset = path.parent / filename
            if asset.resolve().parent != path.parent:
                raise ValueError("qualification artifacts must remain beside the report")
            if hashlib.sha256(asset.read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError(f"qualified {name} hash mismatch")
            assets[name] = asset
        return cls(report, assets)
