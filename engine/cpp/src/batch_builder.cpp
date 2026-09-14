/**
 * Batch building is implemented in iteration_loop.cpp:
 *   build_decode_batch()/build_prefill_batch() write into persistent CPU staging.
 *   copy_batch() transfers visible metadata slices on the copy stream.
 *   step() orders copies and consumers with events.
 *
 * On mixed iterations, prefill construction runs after decode launch, overlapping
 * CPU work with GPU decode. Cross-iteration construction remains future work:
 * step() reads sampled tokens before returning to the scheduler.
 */

#include "iteration_loop.hpp"
