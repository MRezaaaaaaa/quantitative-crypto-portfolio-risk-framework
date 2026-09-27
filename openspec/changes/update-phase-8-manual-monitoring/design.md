# Manual monitoring construction boundary

`ManualMonitoringRecipe` contains explicit weights, risk settings, cash policy
and source provenance only. It contains no optimizer or simulation recipe.
New `ExperimentCreationWorkflow` requests require this recipe. Manual snapshot
construction never invokes an optimizer or learns weights from prices.

Use the existing snapshot/table structure to preserve local databases: a manual
snapshot has objective `manual`, solver `none`, and status `manual_validated`.
Activation requires matching manual provenance and validated long-only weights,
launch prices, values and quantities. Existing optimized records retain their
original fields and recipes. Live parsing accepts both historical recipe formats.
Legacy replay remains available for already-recorded optimized recipes; the new
creation workflow and UI expose no optimized creation path.

Internal date fields retain `training_*` and `optimization_as_of` for database
compatibility. In manual UI they mean risk-history start/cutoff and allocation
decision date, not optimizer inputs. Snapshot hashing includes only declared
history and launch prices, not later evaluation prices. Replay remains sequential.

Daily monitoring retains fixed quantities, drifted weights, origin-safe VaR/ES,
and matured loss evaluation. No fabricated expected-return forecast is recorded.
