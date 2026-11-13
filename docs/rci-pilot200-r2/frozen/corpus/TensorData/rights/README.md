# TensorData asset-rights manifests

This directory records the rights and licensing basis for datasets, source evidence, model artifacts and derived assets. It is separate from the software license of this repository.

## Required location

Place one JSON document per asset or independently licensed asset group in `rights/manifests/`. Use the schema in `rights/manifest.schema.json`. Do not put private credentials or confidential contracts in Git; store an access-controlled evidence reference and checksum instead.

## Status semantics

- `unknown`: rights have not been established; quarantine.
- `needs_review`: evidence exists but has not been approved; quarantine.
- `restricted`: use is limited to explicitly authorized purposes; enforce those restrictions.
- `approved`: a named reviewer approved the documented purposes based on evidence.
- `prohibited`: do not use, train on, publish or redistribute.

`NOASSERTION` is a statement of uncertainty, never a license grant. A permissive license string without evidence of source authority is not sufficient for approval.

## Release rule

Only `approved` assets can be promoted for the exact purposes listed in `permitted_purposes`. Redistribution, commercial use and model training are independent permissions and must be recorded independently. A downstream dataset must preserve the restrictive obligations of its inputs unless a reviewed legal basis says otherwise.

## Validation

Run:

```bash
python scripts/check_rights_manifests.py
```

The GitHub Actions workflow runs this check on changes. The validator checks structure and minimum approval evidence; it does not determine legal validity. Human review remains mandatory for high-impact, ambiguous, personal, contractual or third-party material.
