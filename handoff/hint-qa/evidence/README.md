# Evidence serialization

These compact reports retain measured numerical values, source/input SHA-256 hashes and rejection/timeout outcomes. Machine-specific paths and host identifiers were replaced by logical source labels; serialized report bytes therefore differ from the original private execution logs. `HINT_CHECKOUT`, `VMEX_CHECKOUT`, `SOLVAX_CHECKOUT`, and `QA_INPUTS` are provenance labels, not required directories. `UNBUNDLED_ARTIFACT` and `UNBUNDLED_ENVIRONMENT` explicitly identify records not included here.

Full HINT restart trajectories and installed binaries are not bundled. Historical file hashes identify them but do not make them retrievable. Reproduce a fresh run using the original input fixtures and the patched maintained HINT source; do not claim to resume an absent restart. CPU/GPU backend versions, solver source hashes and actual-primal certificates are retained in the root and derivative reports.

The calibration report records a timeout, not success. The HINT startup refinement pair was still running when initial handoff assembly began; its final status must be read from the main handoff and subsequent evidence, not inferred from input decks.
