.PHONY: list smoke run validate

PYTHON ?= python3
export PYTHONPATH := $(CURDIR)/src:$(CURDIR)/legacy:$(CURDIR):$(PYTHONPATH)

list:
	$(PYTHON) -m ncfusion list

smoke:
	$(PYTHON) -m ncfusion smoke --output results/smoke

run:
	$(PYTHON) -m ncfusion run $(EXPERIMENT) \
		$(if $(BENCHMARK),--benchmark $(BENCHMARK),) \
		$(if $(METHOD),--method $(METHOD),) \
		--output $(or $(OUTPUT),results/runs/$(EXPERIMENT))

validate:
	$(PYTHON) -m ncfusion validate $(ACTUAL) \
		--reference $(or $(REFERENCE),results/reference/table4.csv)
