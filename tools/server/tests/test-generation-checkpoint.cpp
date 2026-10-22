#include "server-generation-checkpoint.h"
#include <cassert>
int main() {
    llama_context target{{1,2,3}}, draft{{4,5,6}};
    server_generation_checkpoint c;
    assert(server_generation_checkpoint_capture(&target, &draft, 2, 100, "test", c));
    assert(c.valid() && c.size_bytes() == 6);
    target.state = {7,8,9};
    draft.state = {10,11,12};
    assert(server_generation_checkpoint_restore_kv(&target, &draft, 2, c));
    assert((target.state == std::vector<uint8_t>{1,2,3}));
    assert((draft.state == std::vector<uint8_t>{4,5,6}));
    target.state = {9};
    assert(!server_generation_checkpoint_restore_kv(&target, &draft, 3, c));
    assert((target.state == std::vector<uint8_t>{9}));
    draft.fail_write = true;
    assert(!server_generation_checkpoint_restore_kv(&target, &draft, 2, c));
    assert((target.state == std::vector<uint8_t>{9}));
    assert(!server_generation_checkpoint_restore_kv(&target, nullptr, 2, c));
    llama_context single{{42}};
    server_generation_checkpoint single_checkpoint;
    assert(server_generation_checkpoint_capture(&single, nullptr, 0, 1, "single", single_checkpoint));
    single.state = {99};
    assert(server_generation_checkpoint_restore_kv(&single, nullptr, 0, single_checkpoint));
    assert(single.state[0] == 42);
}
