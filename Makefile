.PHONY: docs docs-clean docs-open

# Generate API docs with pdoc, excluding the legacy/ folder and the unimplemented leaky integrator.
docs:
	uv run pdoc \
		./models \
		./systems \
		./parameters/duffing_oscillator.py \
		./parameters/hyperparameters.py \
		./training \
		./utils \
		./config.py \
		./main.py \
		--output-dir docs \
		--search \
		--math \
		--mermaid \
		--no-show-source \
		--docformat google

docs-clean:
	rm -rf docs

docs-open:
	open docs/index.html || true

