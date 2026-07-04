"""Tests for JAX compile-log summaries used by performance diagnostics."""

from __future__ import annotations

from tools.diagnostics import summarize_jax_compile_log as summary


def test_compile_name_counts_extracts_jit_names() -> None:
    """The parser should count JAX `Compiling jit(NAME)` entries only."""

    counts = summary.compile_name_counts(
        "\n".join(
            [
                "Compiling jit(_run_scan) with global shapes and types ...",
                "Finished XLA compilation of jit(_run_scan) in 1.0 sec",
                "Compiling jit(broadcast_in_dim) with global shapes and types ...",
                "Compiling jit(_run_scan) with global shapes and types ...",
            ]
        )
    )

    assert counts["_run_scan"] == 2
    assert counts["broadcast_in_dim"] == 1
    assert "Finished XLA compilation" not in counts


def test_summarize_compile_log_and_markdown(tmp_path) -> None:
    """Compile-log summaries should be deterministic and renderable."""

    log = tmp_path / "jax.log"
    log.write_text(
        "\n".join(
            [
                "Compiling jit(multiply) with global shapes and types ...",
                "Compiling jit(_run_scan) with global shapes and types ...",
                "Compiling jit(multiply) with global shapes and types ...",
            ]
        ),
        encoding="utf-8",
    )

    payload = summary.summarize_compile_log(log, top=1)
    text = summary.render_markdown(payload)

    assert payload["total_compile_events"] == 3
    assert payload["unique_compile_names"] == 2
    assert payload["kernels"] == [{"name": "multiply", "count": 2}]
    assert "total compile events: 3" in text
    assert "| multiply | 2 |" in text
