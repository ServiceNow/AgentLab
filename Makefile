.PHONY: test setup miniwob lint stop-miniwob osworld

setup:
	@pip install -e .
	@playwright install chromium --with-deps
	@python -c 'import nltk; nltk.download("punkt_tab")'

miniwob: stop-miniwob
	@git clone https://github.com/Farama-Foundation/miniwob-plusplus.git || true
	@cd miniwob-plusplus && git checkout 7fd85d71a4b60325c6585396ec4f48377d049838
	@python -m http.server 8080 --directory miniwob-plusplus/miniwob/html & echo $$! > .miniwob-server.pid
	@sleep 3
	@echo "MiniWob server started on http://localhost:8080"

check-miniwob:
	@curl -I "http://localhost:8080/miniwob/" || (echo "MiniWob not reachable" && exit 1)
	@echo "MiniWob server is reachable"

stop-miniwob:
	@kill -9 `cat .miniwob-server.pid` || true
	@rm -f .miniwob-server.pid
	@echo "MiniWob server stopped"

run-tests:
	@MINIWOB_URL="http://localhost:8080/miniwob/" pytest -n 5 --durations=10 -m 'not pricy' tests/
	@echo "Tests completed"

test: setup miniwob check-miniwob run-tests stop-miniwob

lint: setup
	@black src/ --check --diff
	@darglint -v 2 -z short src/

osworld:
	@echo "Setting up OSWorld..."
	@git clone https://github.com/xlang-ai/OSWorld || true
	@echo "Modifying OSWorld requirements.txt to remove pinned versions..."
	@cd OSWorld && \
		sed -i.bak 's/numpy~=.*/numpy/' requirements.txt && \
		sed -i.bak 's/torch~=.*/torch/' requirements.txt && \
		sed -i.bak 's/torch$$/torch/' requirements.txt && \
		sed -i.bak 's/tqdm~=.*/tqdm/' requirements.txt && \
		sed -i.bak 's/pandas~=.*/pandas/' requirements.txt
	@echo "Installing OSWorld requirements..."
	@cd OSWorld && pip install -r requirements.txt
	@echo "Installing OSWorld in development mode..."
	@cd OSWorld && pip install -e .
	@echo "OSWorld setup completed!"
	@echo "Next steps:"
	@echo "1. Configure your VM (VMware/VirtualBox) according to OSWorld documentation"
	@echo "2. Download or set up the Ubuntu VM image"
	@echo "3. Run AgentLab with OSWorld tasks"