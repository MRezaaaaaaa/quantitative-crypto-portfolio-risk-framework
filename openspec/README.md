# OpenSpec Governance

This directory records the current behavioral contracts and the reviewed
history of specification changes. The repository currently validates with
OpenSpec CLI `0.16.0`.

## Layout

```text
openspec/
├── specs/
│   └── <capability>/spec.md
└── changes/
    ├── <active-change>/
    └── archive/
        └── YYYY-MM-DD-<change>/
```

- `specs/` is current truth for implemented behavior and explicit public
  boundaries.
- `changes/` is reserved for proposed behavioral work.
- `changes/archive/` preserves completed historical records; archived records
  are not rewritten to match current terminology.

## Validation

Install the reviewed CLI version and validate every active capability:

```bash
npm install --global @fission-ai/openspec@0.16.0
openspec list --specs
openspec validate --all --strict --no-interactive
```

Every current capability MUST use:

```text
openspec/specs/<capability>/spec.md
```

Every normative `### Requirement:` MUST include at least one
`#### Scenario:`. Requirement text uses `SHALL` or `MUST`; scenarios state
observable `WHEN` and `THEN` behavior.

## Change control

New behavior, architecture, model assumptions, or public API changes require a
verb-led change directory with a proposal, tasks, and capability delta specs.
Implementation starts only after the proposal and model-risk implications are
reviewed. Typographical and non-behavioral documentation corrections may be
applied directly.

An archived change is historical evidence, not proof that its original claims
remain current. The active specs and tested code are authoritative.
