.PHONY: list smoke run validate

PYTHON ?= python3
export PYTHONPATH := $(CURDIR)/src:$(CURDIR)/legacy:$(CURDIR):$(PYTHONPATH)

list:
	$(PYTHON) -m ncfusion list

smoke:
	$(PYTHON) -m ncfusion smoke --output micro_artifact/results/smoke

run:
	$(PYTHON) -m ncfusion run $(EXPERIMENT) \
		$(if $(BENCHMARK),--benchmark $(BENCHMARK),) \
		$(if $(METHOD),--method $(METHOD),) \
		--output $(or $(OUTPUT),micro_artifact/results/runs/$(EXPERIMENT))

validate:
	$(PYTHON) -m ncfusion validate $(ACTUAL) \
		--reference $(or $(REFERENCE),micro_artifact/results/reference/table4.csv)
