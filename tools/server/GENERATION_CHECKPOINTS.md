# Experimental generation checkpoint backend

This branch adds a real, explicit **sequence state snapshot** primitive using
`llama_state_seq_get_size_ext` and `llama_state_seq_get_data_ext`.
It is not a constant-time token-position bookmark: the state is copied.

The branch also provides `server_generation_checkpoint_restore_kv()` for a guarded
round-trip of the target and optional draft **KV sequence state only**. It rejects
sequence mismatches and model-pair mismatches, snapshots both live states before
writing, and attempts to recover the pre-call states on failure. A failed
restore must be treated as a poisoned slot and reinitialized; best-effort
recovery is not a transactional guarantee. Neither capture nor restore is
wired to phase markers or a public HTTP route.

The server slot has `capture_generation_checkpoint(checkpoint_id, token_position)`,
which captures both target and optional draft contexts, stores them under a
slot-scoped identifier, and logs either:

```text
checkpoint captured: id=... position=... bytes=... (KV snapshot only; restore not enabled)
checkpoint capture failed: id=... position=...
```

Capture is **not yet remotely callable**. No checkpoint is created automatically.
The argument `token_position` is metadata supplied by the caller and must be
derived from the actual logical context by the eventual control API.

## Safety invariants for the next iteration

1. Pause the owning decode slot and drain any in-flight batch first.
2. Capture and restore authoritative prompt tokens, generated token history,
   sampler state, sequence KV state, and draft-model state as one coherent snapshot.
3. Reject checkpoint IDs from other slots and invalidate snapshots on slot reset
   or incompatible model/revision.
4. Ensure stable chat-message boundaries when injecting replacement text.
5. Never confuse rollback of context with rollback of filesystem side effects.
6. Preserve default inference behaviour when checkpoint control is not used.

## Pending

- Authorized checkpoint API bound to one live slot/request, with generation pause.
- Complete state snapshot, including sampler and token sequence.
- Safe restore, append and resume operations with strict validation.
- Integration tests on a compiled llama-server and several chat templates.

This is a development branch and should not be merged as a functional
checkpoint-restore feature yet.

## Experimental control workflow

Checkpoint controls are routed through the existing POST /control endpoint using the active completion ID. Available actions are checkpoint_pause, checkpoint_save, checkpoint_restore, checkpoint_resume. Save and restore require a non-empty checkpoint_id. Pause the slot before either operation; resume explicitly afterward. All checkpoints are opt-in and tied to a single active completion.

Current implementation rejects streaming and speculative completions; speculative controller and externally streamed bytes cannot be rewound safely. Its snapshot saves the prompt, cloned sampler, pending sampled token, generated text and token history, stopping state, and KV state together. Only four snapshot IDs per slot are retained. Checkpoints disappear upon slot reset.

This is not a complete harness integration. There is no token injection, no filesystem rollback, and no evidence of a successful compiled llama-server run. Integration tests still must cover pause/save/restore/resume, concurrent slots, cancellation, stop strings, context shift, and sampler replay.

Standalone mock check (not a server integration test):

    c++ -std=c++17 -Wall -Wextra -pedantic -Itools/server/tests/generation-checkpoint-mock -Itools/server tools/server/tests/test-generation-checkpoint.cpp -o /tmp/test-generation-checkpoint
    /tmp/test-generation-checkpoint
