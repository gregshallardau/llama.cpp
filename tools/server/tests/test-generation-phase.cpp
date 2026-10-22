// Standalone smoke test:
// c++ -std=c++17 -Wall -Wextra -pedantic tools/server/tests/test-generation-phase.cpp -o /tmp/test-generation-phase && /tmp/test-generation-phase
#include "../server-generation-phase.h"

#include <cassert>
#include <string>
#include <vector>

int main() {
    server_generation_phase_tracker tracker({
        {"<think>", server_generation_phase::reasoning, false},
        {"</think>", server_generation_phase::reasoning, true},
        {"<tool_call>", server_generation_phase::tool_call, false},
        {"</tool_call>", server_generation_phase::tool_call, true},
    });

    assert(tracker.append("Hello <thi").empty());
    auto events = tracker.append("nk>reason");
    assert(events.size() == 1);
    assert(events[0].phase == server_generation_phase::reasoning);
    assert(events[0].byte_offset == 6);
    assert(tracker.phase() == server_generation_phase::reasoning);

    events = tracker.append("</think> <tool_call>{}</tool_call>");
    assert(events.size() == 3);
    assert(events[0].phase == server_generation_phase::response);
    assert(events[1].phase == server_generation_phase::tool_call);
    assert(events[2].phase == server_generation_phase::response);
    assert(tracker.phase() == server_generation_phase::response);

    // Marker configuration is model-specific. Empty configuration is passive.
    server_generation_phase_tracker no_markers({});
    assert(no_markers.append("text <tool_call>") .empty());
    assert(no_markers.phase() == server_generation_phase::response);

    return 0;
}
