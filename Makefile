# Convenience wrapper around the venv. Every target uses the pinned interpreter
# so behaviour does not depend on whatever Python happens to be on PATH.
PY := ./.venv/bin/python
PIP := ./.venv/bin/pip
# Run the package straight from src/ — no reliance on an editable install being
# present, which keeps `make` working from a fresh checkout.
RUN := PYTHONPATH=src ./.venv/bin/python

.PHONY: setup test smoke binary binary-loso fourclass clean help

help:
	@echo "make setup        - create .venv (python3.12) and install pinned deps"
	@echo "make test         - run unit tests"
	@echo "make smoke        - end-to-end run on ~3-5 subjects (fast sanity check)"
	@echo "make binary       - full within-subject binary benchmark (heavy)"
	@echo "make binary-loso  - subject-independent (LOSO) binary benchmark (heavy)"
	@echo "make fourclass    - 4-class experiment (heavy)"
	@echo "make clean        - remove caches and generated artifacts"

setup:
	/opt/homebrew/bin/python3.12 -m venv .venv
	$(PIP) install --upgrade pip wheel
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

test:
	$(PY) -m pytest

smoke:
	$(RUN) -m mibci.run --config configs/smoke.yaml --experiment binary

binary:
	$(RUN) -m mibci.run --config configs/binary.yaml --experiment binary --cv within

binary-loso:
	$(RUN) -m mibci.run --config configs/binary.yaml --experiment binary --cv loso

fourclass:
	$(RUN) -m mibci.run --config configs/fourclass.yaml --experiment fourclass --cv within

clean:
	rm -rf artifacts/cache artifacts/*.csv artifacts/*.png artifacts/*.json
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
