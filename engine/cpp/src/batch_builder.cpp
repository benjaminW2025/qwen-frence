/**
 * Batch building is implemented in iteration_loop.cpp:
 *   build_batch() writes into persistent CPU staging (pinned for CUDA).
 *   copy_batch() transfers active metadata slices on the copy stream.
 *   step() orders copies and consumers with events.
 *
 * Cross-iteration batch construction remains future work: step() currently
 * reads sampled tokens before returning to the scheduler.
 */

#include "iteration_loop.hpp"
