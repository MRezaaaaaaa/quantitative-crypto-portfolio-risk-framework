# Dependency Management

## Source of truth

`pyproject.toml` is the only authoritative list of direct dependencies and
supported version ranges.

- Core runtime packages are declared in `project.dependencies`.
- Streamlit is isolated in the `app` optional dependency group.
- Test, coverage, build, and lint tools are isolated in the `dev` group.
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

No lockfile is committed yet. Creating a lock from an unrelated global Python
environment would capture platform-specific and extraneous packages and would
create false reproducibility.

Before `v1.0.0`, adopt one reviewed cross-platform lock workflow, generate the
lock from `pyproject.toml`, and verify it on every supported Python version. The
chosen lock must record solver and scientific-computing dependencies without
replacing `pyproject.toml` as the package metadata source.

Until that decision is implemented, CI clean installation across Python 3.10 to
3.13 is the compatibility gate; it is not equivalent to a fully locked research
environment.
