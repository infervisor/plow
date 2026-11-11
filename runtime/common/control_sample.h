#pragma once
#include <stddef.h>
#include <stdint.h>

struct ControlSamplePlan {
    uint64_t seed;
    float temperature, top_p, presence_penalty, stop_p, stop_hazard, period_thr, boost;
    uint32_t top_k, penalty_window, min_frames, cap, stall, period_win, pmin, pmax;
    uint32_t start, every, backstop, hist, n_windows;
    uint32_t windows[16][2];
};
struct ControlSampleState {
    ControlSamplePlan plan;
    uint32_t ticks, frames, last, boost_at, backstop, stopped;
    float hazard, aeos_bias;
    int32_t recent[256], tail[256];
};
struct ControlSampleRow { uint32_t slot, output, logits; };
static_assert(sizeof(ControlSamplePlan) == 216, "controlled sampler plan ABI");
static_assert(sizeof(ControlSampleState) == 2296, "controlled sampler state ABI");
static_assert(sizeof(ControlSampleRow) == 12, "controlled sampler row ABI");
static_assert(alignof(ControlSampleState) == 8, "controlled sampler state alignment");
static_assert(offsetof(ControlSampleState, last) == 224, "controlled sampler cb0 ABI");
static_assert(offsetof(ControlSampleState, stopped) == 236, "controlled sampler stop ABI");
