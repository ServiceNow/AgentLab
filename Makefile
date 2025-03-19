.PHONY: test setup miniwob lint stop-miniwob

setup:
	@pip install -e .
	@playwright install chromium --with-deps
	@python -c 'import nltk; nltk.download("punkt_tab")'

miniwob: stop-miniwob
	@git clone https://github.com/Farama-Foundation/miniwob-plusplus.git || true
	@cd miniwob-plusplus && git checkout 7fd85d71a4b60325c6585396ec4f48377d049838
	@python -m http.server 8080 --directory miniwob-plusplus/miniwob/html & echo $$! > .miniwob-server.pid
	@echo "MiniWob server started on http://localhost:8080"

stop-miniwob:
	@kill -9 `cat .miniwob-server.pid` || true
	@rm -f .miniwob-server.pid
	@echo "MiniWob server stopped"

run-tests:
	@MINIWOB_URL="http://localhost:8080/miniwob/" pytest -n 5 --durations=10 -m 'not pricy' tests/
	@echo "Tests completed"

test: setup miniwob run-tests stop-miniwob

lint: setup
	@black src/ --check --diff
	@darglint -v 2 -z short src/
