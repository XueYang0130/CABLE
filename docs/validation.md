# Release validation

Validated on 8 November 2026 before publication:

- 134 tests passed.
- All 175 archived configurations parsed successfully.
- Average performance and forgetting matched the previously audited values for every result group.
- Two-task, 400-step environment smoke run completed, including SAC updates and evaluations. This short run did not produce successful-episode BC memory; reservoir, relabeling, and gradient-coordination behavior are covered by the tests.
- Learning curves regenerated from the included evaluation records.
- No local absolute paths, Chinese text, or credential-shaped strings were found in release source/configuration/evidence files.

This validates the packaged implementation and result processing, not a fresh full retraining of every reported experiment. JSON configuration loading explicitly uses the JSON parser to preserve scientific-notation numeric types.
