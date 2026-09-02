PYTHON ?= python3
VLLM_PYTHON ?= /root/vllm-bench-env/bin/python
ifndef RUN_ID
RUN_ID := $(shell date -u +%Y%m%dT%H%M%SZ)
endif
ARTIFACT_DIR ?= artifacts/reproduction/$(RUN_ID)
LOCAL_SUITE ?=

.NOTPARALLEL: reproduce-local reproduce-vllm reproduce-experiments

.PHONY: help unit preflight-local preflight-vllm correctness correctness-primary \
	benchmark-smoke phase-scorecard intervention-suite dispatch-policy-suite \
	scorecard-local scorecard-vllm reproduce-local reproduce-vllm reproduce-experiments

help:
	@echo "Reproducible validation and benchmark targets"
	@echo "  make unit                 CPU-only harness contracts"
	@echo "  make preflight-local      H100/local-engine setup and 1-request smoke"
	@echo "  make correctness          complete GPU correctness suite"
	@echo "  make correctness-primary  eager production checks; skips legacy graphs"
	@echo "  make benchmark-smoke      tiny local control/candidate benchmark"
	@echo "  make phase-scorecard      isolated prefill and decode surfaces"
	@echo "  make intervention-suite   focused operator/intervention ablations"
	@echo "  make dispatch-policy-suite  dense crossover sweeps and policy fit"
	@echo "  make scorecard-local      complete eight-cell local H100 scorecard"
	@echo "  make scorecard-vllm LOCAL_SUITE=... VLLM_PYTHON=..."
	@echo "  make reproduce-local      unit + preflight + correctness + local scorecard"
	@echo "  make reproduce-vllm LOCAL_SUITE=...  preflight + matched vLLM replay"
	@echo "  ARTIFACT_DIR defaults to $(ARTIFACT_DIR)"

unit:
	$(PYTHON) -m unittest discover -s benchmarks/tests -p 'test_*.py'
	$(PYTHON) -m unittest discover -s correctness/tests -p 'test_*.py'
	$(PYTHON) -m unittest discover -s experiments/tests -p 'test_*.py'

preflight-local:
	$(PYTHON) benchmarks/run_setup_checks.py \
		--suite local \
		--json-out $(ARTIFACT_DIR)/preflight-local.json

preflight-vllm:
	$(VLLM_PYTHON) benchmarks/run_setup_checks.py \
		--suite vllm \
		--json-out $(ARTIFACT_DIR)/preflight-vllm.json

correctness:
	$(PYTHON) correctness/run_correctness.py \
		--checks all \
		--json-out $(ARTIFACT_DIR)/correctness.json

correctness-primary:
	$(PYTHON) correctness/run_correctness.py \
		--checks baseline \
		--json-out $(ARTIFACT_DIR)/correctness-primary.json

benchmark-smoke:
	$(PYTHON) benchmarks/run_benchmarks.py \
		--backends custom-kernels,regime-dispatched \
		--workload-name reproduction-smoke \
		--num-requests 2 \
		--prompt-lengths 128 \
		--output-lengths 8 \
		--max-running 2 \
		--max-num-batched-tokens 512 \
		--warmups 1 \
		--repetitions 1 \
		--seed 0 \
		--strict-backends \
		--output-dir $(ARTIFACT_DIR)/smoke

phase-scorecard:
	$(PYTHON) benchmarks/run_phase_sweep.py \
		--implementations custom-kernels,regime-dispatched \
		--phases prefill \
		--batch-sizes 1,2,4,8 \
		--prefill-lengths 128,512,1024,2048,4096,8192 \
		--warmups 2 \
		--repetitions 5 \
		--output-dir $(ARTIFACT_DIR)/phase-prefill
	$(PYTHON) benchmarks/run_phase_sweep.py \
		--implementations custom-kernels,regime-dispatched \
		--phases decode \
		--batch-sizes 1,4,8,16,32,64 \
		--context-lengths 128,512,1024,2048,4096,8192,16384 \
		--decode-steps 64 \
		--warmups 2 \
		--repetitions 5 \
		--output-dir $(ARTIFACT_DIR)/phase-decode

intervention-suite:
	$(PYTHON) experiments/run_intervention_suite.py \
		--fail-fast \
		--output-dir $(ARTIFACT_DIR)/interventions

dispatch-policy-suite:
	$(PYTHON) experiments/run_dispatch_policy_experiment.py \
		--fail-fast \
		--output-dir $(ARTIFACT_DIR)/dispatch-policy

scorecard-local:
	$(PYTHON) benchmarks/run_regime_scorecard.py \
		--mode local \
		--warmups 1 \
		--repetitions 3 \
		--seed 0 \
		--fail-fast \
		--output-dir $(ARTIFACT_DIR)/regime-scorecard

scorecard-vllm:
	@test -n "$(LOCAL_SUITE)" || \
		(echo "LOCAL_SUITE is required; use the suite path printed by scorecard-local"; exit 2)
	$(VLLM_PYTHON) benchmarks/run_regime_scorecard.py \
		--mode vllm \
		--local-suite "$(LOCAL_SUITE)" \
		--warmups 1 \
		--repetitions 3 \
		--seed 0 \
		--fail-fast

reproduce-local: unit preflight-local correctness scorecard-local

reproduce-vllm: preflight-vllm scorecard-vllm

reproduce-experiments: phase-scorecard intervention-suite dispatch-policy-suite
