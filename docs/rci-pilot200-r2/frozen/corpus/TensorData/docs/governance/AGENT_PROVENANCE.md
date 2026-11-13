# Agent Execution Provenance

TensorData records agent participation as provenance metadata, not as a legal signature or approval.

## API

The `POST /api/v1/provenance` request accepts an optional `actor` object:

```json
{
  "activity_type": "dataset_transform",
  "inputs": [{"artifact_id": "INPUT_ARTIFACT_ID", "sha256": "INPUT_SHA256"}],
  "outputs": [{"artifact_id": "OUTPUT_ARTIFACT_ID", "sha256": "OUTPUT_SHA256"}],
  "parameters": {"transform": "normalize_units"},
  "responsible_actor": "agent:physics-data-curator-03",
  "actor": {
    "actor_type": "agent",
    "actor_id": "physics-data-curator-03",
    "role": "producer",
    "tool_name": "Claude Code",
    "tool_version": "VERSION_IF_KNOWN",
    "model_id": "MODEL_IF_KNOWN",
    "run_id": "RUN_OR_SESSION_REFERENCE",
    "work_package": "WP-123",
    "accountable_human": "project-maintainer",
    "reviewer_agent_ids": ["independent-reviewer-02"],
    "verification_refs": ["https://github.com/OWNER/REPO/actions/runs/123"]
  }
}
```

Replace illustrative values with actual values. If a value is not exposed, omit it; never invent it.

## Semantics

- `responsible_actor` is the readable actor label used by existing clients.
- `actor.actor_type` distinguishes human, agent, system and unknown actors.
- Agent records require `actor_id`, `tool_name` and `accountable_human`.
- `reviewer_agent_ids` and `verification_refs` record quality evidence. They do not represent human approval.
- The API stores the actor object under the reserved `parameters_json._actor_provenance` key. Clients cannot override this key through free-form `parameters`.
- Legacy requests without `actor` remain accepted for compatibility but are explicitly marked `unknown / legacy_unattributed`; they are not silently attributed to a human.
- No API field in this initial version can assert a trusted human legal approval. Human approval requires a separately authenticated, auditable workflow and must not be asserted by an agent using the shared service API key.

## Operational rule

For new agent-generated data, code-derived datasets, model artifacts and transformations, send actor metadata on every provenance record. Keep the accountable human identifier stable and non-personal (for example, `project-maintainer`). Link to task/PR/commit/test evidence rather than storing private prompts, credentials or internal reasoning.

This is an attribution and auditability improvement, not proof that the agent output is original, copyrightable, or legally licensable.
