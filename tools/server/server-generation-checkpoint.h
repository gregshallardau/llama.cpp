// Experimental sequence state snapshot. Not automatically enabled or restored.
// A snapshot is a serialized copy, not a cheap position marker; the backend
// must coordinate generation state, token history, sampler and prompt on restore.
#pragma once

#include "llama.h"

#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

struct server_generation_checkpoint {
    std::string id;
    llama_seq_id seq_id = -1;
    int64_t logical_token_position = -1;
    std::vector<uint8_t> target_state;
    std::vector<uint8_t> draft_state;

    size_t size_bytes() const {
        return target_state.size() + draft_state.size();
    }
    bool valid() const {
        return seq_id >= 0 && !target_state.empty();
    }
};

inline bool server_generation_checkpoint_capture_sequence(
        llama_context * ctx, llama_seq_id seq_id, std::vector<uint8_t> & out) {
    if (!ctx || seq_id < 0) return false;
    const size_t size = llama_state_seq_get_size_ext(ctx, seq_id, LLAMA_STATE_SEQ_FLAGS_NONE);
    if (size == 0 || size > std::numeric_limits<size_t>::max() / 2) return false;
    std::vector<uint8_t> candidate(size);
    const size_t copied = llama_state_seq_get_data_ext(
        ctx, candidate.data(), candidate.size(), seq_id, LLAMA_STATE_SEQ_FLAGS_NONE);
    if (copied == 0 || copied > candidate.size()) return false;
    candidate.resize(copied);
    out = std::move(candidate);
    return true;
}

// Captures both target and optional draft model states. Transactional in the
// sense that the caller's previous snapshot is unchanged if capture fails.
// Must be invoked from the owning decode thread when the slot is quiescent.
// Does not change inference state or attempt to restore it.
inline bool server_generation_checkpoint_capture(
        llama_context * target, llama_context * draft, llama_seq_id seq_id,
        int64_t token_position, const std::string & id,
        server_generation_checkpoint & output) {
    server_generation_checkpoint candidate;
    candidate.id = id;
    candidate.seq_id = seq_id;
    candidate.logical_token_position = token_position;
    if (!server_generation_checkpoint_capture_sequence(target, seq_id, candidate.target_state)) {
        return false;
    }
    if (draft && !server_generation_checkpoint_capture_sequence(draft, seq_id, candidate.draft_state)) {
        return false;
    }
    output = std::move(candidate);
    return true;
}

// Restores only the llama sequence payloads. This deliberately does NOT resume
// generation: the owner must separately restore the prompt, sampler, generated
// text/tokens, speculative controller, and streaming cursor from the same boundary.
// The owning decode thread must be quiescent and no batch may be in flight.
// If a write fails, attempt to restore both pre-call snapshots. Since a backend
// may reject rollback too, callers must abort/reinitialize the slot on failure.
inline bool server_generation_checkpoint_restore_sequence(
        llama_context * ctx, llama_seq_id seq_id, const std::vector<uint8_t> & state) {
    return ctx && !state.empty() &&
        llama_state_seq_set_data_ext(ctx, state.data(), state.size(), seq_id,
                                     LLAMA_STATE_SEQ_FLAGS_NONE) == state.size();
}

inline bool server_generation_checkpoint_restore_kv(
        llama_context * target, llama_context * draft, llama_seq_id seq_id,
        const server_generation_checkpoint & snapshot) {
    if (!snapshot.valid() || !target || seq_id < 0 || snapshot.seq_id != seq_id ||
        (draft == nullptr) != snapshot.draft_state.empty()) {
        return false;
    }

    // Capture the live state before touching either context.
    std::vector<uint8_t> previous_target;
    std::vector<uint8_t> previous_draft;
    if (!server_generation_checkpoint_capture_sequence(target, seq_id, previous_target) ||
        (draft && !server_generation_checkpoint_capture_sequence(draft, seq_id, previous_draft))) {
        return false;
    }

    if (server_generation_checkpoint_restore_sequence(target, seq_id, snapshot.target_state) &&
        (!draft || server_generation_checkpoint_restore_sequence(draft, seq_id, snapshot.draft_state))) {
        return true;
    }

    // Best-effort recovery. A failure still leaves the slot unusable until
    // its owner resets it, even if these calls appear to succeed.
    server_generation_checkpoint_restore_sequence(target, seq_id, previous_target);
    if (draft) {
        server_generation_checkpoint_restore_sequence(draft, seq_id, previous_draft);
    }
    return false;
}
