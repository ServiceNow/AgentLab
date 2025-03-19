.PHONY: test setup miniwob lint
setup:
	@pip install darglint black
	@pip install -e .
	@playwright install chromium --with-deps
	@python -c 'import nltk; nltk.download("punkt_tab")'

miniwob:
	@kill -9 `cat .miniwob-server.pid` || true
	@git clone https://github.com/Farama-Foundation/miniwob-plusplus.git || true
	@cd miniwob-plusplus && git checkout 7fd85d71a4b60325c6585396ec4f48377d049838
	@python -m http.server 8080 --directory miniwob-plusplus/miniwob/html & echo $$! > .miniwob-server.pid
	@sleep 2
	@echo "MiniWob server started on port 8080"
	@echo "To stop the server: kill \`cat .miniwob-server.pid\`"

test: setup miniwob
	@MINIWOB_URL="http://localhost:8080/miniwob/" pytest -n 5 --durations=10 -m 'not pricy' -v tests/

lint:
	black src/ --check --diff
	darglint -v 2 -z short src/
