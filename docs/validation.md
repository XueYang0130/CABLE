# Release validation

The reader workflow was checked using a fresh clone of the public repository and a new Python 3.11 virtual environment on macOS.

- Installation from `requirements.txt` succeeded; `pip check` reported no dependency conflicts.
- 137 tests passed, including gradient coordination, reference memory, baseline behavior, and entry-point checks.
- All 175 archived configurations parsed successfully.
- Average performance and forgetting were recomputed from the included evaluation records and matched the archived aggregate results exactly.
- The learning-curve plotting command generated PDF and PNG outputs.
- Short training and evaluation runs completed for CABLE, fine-tuning, ClonEx-SAC, RECALL, PackNet, a CW3 sequence, and a single-task batch. These reduced-budget checks do not establish full-budget performance.

Reader-facing fixes include seed-specific output names, configuration-path handling, a standalone reference-aggregation entry point, and method defaults matching the paper configurations. The `--smoke` option uses a reduced budget to check execution; it does not reproduce paper performance.

CPU execution was checked. CUDA, MPS, Windows, and Linux have not been independently validated by this check. Full-budget experiments were not rerun as part of release validation.
