## Setup instructions
- you need podman installed to run code execution and a serper.dev api key to use web search
- to install and configure podman on a mac use provided script `src/agentlab/agents/tapeagent/experiments/setup_gaia.sh`
- after the podman machine up and running set DOCKER_HOST env var to its socket: `export DOCKER_HOST=http+unix://$(podman machine inspect --format '{{.ConnectionInfo.PodmanSocket.Path}}')`
- set the env var with the serper dev api key: `export SERPER_API_KEY=your_key`
- set the env var with the url to the inference endpoint: `export LLM_BASE_URL=your_enpoint_url`

## Experiment configs:
- main config: `src/agentlab/agents/tapeagent/conf/gaia_l1.yaml` for L1 subset, `src/agentlab/agents/tapeagent/conf/gaia_val.yaml` for full validation set
- llm configs are in `src/agentlab/agents/tapeagent/conf/llm`. Feel free to add your own
- recommended agent architecture to use is `src/agentlab/agents/tapeagent/conf/agent/plan_act.yaml`. It is already used in the main configs mentioned above.
- env config that describes available tools: `src/agentlab/agents/tapeagent/conf/environment/web_code.yaml`

## Running evaluation:
- to run in debug mode without parallelism: `AGENTLAB_DEBUG=1 python src/agentlab/agents/tapeagent/experiments/run_gaia.py`
- to run quick parallel eval: `python src/agentlab/agents/tapeagent/experiments/run_gaia.py`
- you can adjust content of the entrypoint script `src/agentlab/agents/tapeagent/experiments/run_gaia.py` to change config name.
- when parallel eval is running, Ray dahsboard with progress is available at `http://127.0.0.1:8265/#/jobs/01000000`
- experiment results will be written in subfolder of '~/agentlab_results/` with the name including current datetime, agent name and benchmark name
