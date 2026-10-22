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
