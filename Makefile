.PHONY: test

test:
	bash -c 'source .venv/bin/activate && LIBTORCH_BYPASS_VERSION_CHECK=1 cargo test --release -- --test-threads=1 2>&1'
