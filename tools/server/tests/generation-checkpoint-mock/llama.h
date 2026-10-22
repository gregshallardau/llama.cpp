#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
using llama_seq_id = int32_t;
enum llama_state_seq_flags { LLAMA_STATE_SEQ_FLAGS_NONE = 0 };
struct llama_context {
    std::vector<uint8_t> state;
    bool fail_write = false;
};
inline size_t llama_state_seq_get_size_ext(llama_context * c, llama_seq_id, llama_state_seq_flags) { return c->state.size(); }
inline size_t llama_state_seq_get_data_ext(llama_context * c, uint8_t * dst, size_t cap, llama_seq_id, llama_state_seq_flags) {
    if (cap < c->state.size()) return 0;
    for (size_t i = 0; i < c->state.size(); ++i) dst[i] = c->state[i];
    return c->state.size();
}
inline size_t llama_state_seq_set_data_ext(llama_context * c, const uint8_t * src, size_t count, llama_seq_id, llama_state_seq_flags) {
    if (c->fail_write) return 0;
    c->state.assign(src, src + count);
    return count;
}
