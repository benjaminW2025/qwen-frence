/**
 * bindings.cpp - Python bindings for C++ iteration loop
 *
 * Usage from Python:
 *   import inference_engine_cpp as cpp
 *
 *   loop = cpp.IterationLoop(config, device)
 *   loop.submit_request([1, 2, 3], max_output_tokens=100)
 *
 *   while loop.num_pending() > 0 or loop.num_running() > 0:
 *       loop.step(forward_fn)
 *       for req_id, output_ids in loop.pop_completed():
 *           print(f"Request {req_id}: {output_ids}")
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <torch/extension.h>

#include "iteration_loop.hpp"

namespace py = pybind11;
using namespace inference_engine;

PYBIND11_MODULE(inference_engine_cpp, m) {
    m.doc() = "C++ iteration loop for inference engine";

    // SchedulerConfig
    py::class_<SchedulerConfig>(m, "SchedulerConfig")
        .def(py::init<>())
        .def_readwrite("max_batch_size", &SchedulerConfig::max_batch_size)
        .def_readwrite("max_prefill_tokens_per_iter", &SchedulerConfig::max_prefill_tokens_per_iter)
        .def_readwrite("max_context_length", &SchedulerConfig::max_context_length)
        .def_readwrite("block_size", &SchedulerConfig::block_size)
        .def_readwrite("num_kv_heads", &SchedulerConfig::num_kv_heads)
        .def_readwrite("head_dim", &SchedulerConfig::head_dim);

    // IterationLoop
    py::class_<IterationLoop>(m, "IterationLoop")
        .def(py::init<SchedulerConfig, torch::Device>(),
             py::arg("config"),
             py::arg("device"))
        .def("submit_request", &IterationLoop::submit_request,
             py::arg("prompt_ids"),
             py::arg("max_output_tokens"),
             "Submit a new request, returns request ID")
        .def("step", &IterationLoop::step,
             py::arg("forward_fn"),
             "Run one iteration, returns number of newly completed requests")
        .def("pop_completed", &IterationLoop::pop_completed,
             "Get and clear completed request outputs")
        .def("num_pending", &IterationLoop::num_pending)
        .def("num_running", &IterationLoop::num_running);

    // ==========================================================================
    // TODO(you): Add any additional bindings you need
    //
    // IDEAS:
    //   - Expose Request class for debugging
    //   - Add stats (tokens/sec, queue depth over time)
    //   - Add preemption control
    //   - Add priority support
    // ==========================================================================
}
