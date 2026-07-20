# Dependency Management

## Source of truth

`pyproject.toml` is the only authoritative list of direct dependencies and
supported version ranges.

- Core runtime packages are declared in `project.dependencies`.
- Streamlit is isolated in the `app` optional dependency group.
- Test, coverage, build, and lint tools are isolated in the `dev` group.
- The build backend and wheel builder appear in both `build-system.requires`
  and the `dev` group so CI can build without a second, unlocked resolver.
- `requirements.txt` and `requirements-dev.txt` are thin compatibility entry
  points; they must not duplicate package versions.

Install the application with:

```bash
python -m pip install -e ".[app]"
```

Install the development environment with:

```bash
python -m pip install -e ".[app,dev]"
```

## Version-range policy

Direct dependencies have tested minimums and conservative upper major-version
bounds. The lower bound describes the oldest intended compatible release; the
upper bound prevents an unreviewed major upgrade from silently changing
numerical or API behavior.

Dependabot may propose upgrades, but Python dependency upgrades remain separate
pull requests so numerical changes can be attributed to a specific package. An
upgrade is accepted only after the complete regression suite, numerical
fixtures, build, and application smoke checks pass.

## Lock strategy

`uv.lock` is the committed, cross-platform snapshot of all direct and
transitive packages, including numerical solvers. It covers the declared
Python support window, 3.10 through 3.13. `pyproject.toml` remains the package
metadata and dependency-policy source; the lockfile must never be edited by
hand.

The lock was generated with uv 0.11.16. Install that version of uv, then create
the exact development environment with:

```bash
uv sync --locked --extra app --extra dev
```

Run commands without allowing an implicit lock update:

```bash
uv run --locked --no-sync python -m pytest -p no:cacheprovider -q
uv run --locked --no-sync ruff check app.py run_demo.py \
  run_phase5_optimization_demo.py src tests scripts
```

`uv lock --check` is the mechanical gate that confirms the lock still matches
`pyproject.toml`. Intentional dependency updates use `uv lock --upgrade-package
<package>` in a dedicated change, followed by the regression and numerical
baseline suites. A plain `uv lock --upgrade` is not part of routine execution.

The thin `requirements*.txt` files remain available for users who prefer pip,
but pip installs resolve within the declared ranges and are therefore
compatibility installs, not the exact research environment.

## CI policy

CI uses the same committed lockfile in every Python 3.10–3.13 job. The lock
check fails if dependency metadata changes without a reviewed lock update.
Dependabot proposals remain separate for Python packages so any change in
financial outputs can be attributed to a specific upgrade.

Release artifacts are built with `python -m build --no-isolation` inside the
locked development environment. Omitting `--no-isolation` would create a second
temporary environment and resolve build requirements independently of
`uv.lock`, weakening artifact reproducibility.
