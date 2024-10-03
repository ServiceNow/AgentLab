

<a href="https://github.com/user-attachments/assets/1bd2f6b2-bce0-43c7-846b-837fd3c6480a">
  <img src="https://github.com/user-attachments/assets/1bd2f6b2-bce0-43c7-846b-837fd3c6480a" width="1000" />
</a>

AgentLab is a framework for developing and evaluating agents on a variety of
benchmarks supported by [BrowserGym](https://github.com/ServiceNow/BrowserGym).
This includes:
* WebArena
* WorkArena.L1, L2, L3
* VisualWebArena (coming soon...)
* MiniWoB

The framework enables the desing of rich hyperparameter spaces and the launch of
parallel experiments using ablation studies or random searches. It also provides
agent_xray, a visualization tool to inspect the results of the experiments using
a custom gradio interface

<a href="https://github.com/user-attachments/assets/20a91e7b-94ef-423d-9091-743eebb4733d">
  <img src="https://github.com/user-attachments/assets/20a91e7b-94ef-423d-9091-743eebb4733d" width="250" />
</a>

## Install agentlab

This repo is intended for testing and developing new agents, hence we clone and install using the `-e` flag.

```bash
git clone git@github.com:ServiceNow/AgentLab.git
pip install -e .
```

## Set Environment Variables

```bash
export AGENTLAB_EXP_ROOT=<root directory of experiment results>  # defaults to $HOME/agentlab_results
export OPENAI_API_KEY=<your openai api key> # if openai models are used
export HUGGINGFACEHUB_API_TOKEN=<your huggingfacehub api token> # if huggingface models are used
```

## Use an assistant to work for you (at your own cost and risk)
```bash
agentlab-assistant --start_url https://www.google.com
```

## Prepare Benchmarks
Depending on which benchmark you use, there are some prerequisites

<details>
<summary>MiniWoB</summary>

```bash
export MINIWOB_URL="file://$HOME/dev/miniwob-plusplus/miniwob/html/miniwob/"
```
</details>

<details>

<summary>WorkArena</summary>

See [detailed instructions on workarena github](https://github.com/ServiceNow/WorkArena?tab=readme-ov-file#getting-started)

At a glance: 
1) [Sign in](https://developer.servicenow.com/) and reqeuest a `washington` instance.
2) Once the instance is ready, you should see `<your instance URL>` and `<your-instance-password>`
3) Add these to your `.bashrc` (or `.zshrc`) and `source` it (note: make sure that
  all variables are in single quotes unless you happen to have a password with a
  single quote in it)
    ```bash
    export SNOW_INSTANCE_URL='https://<your-instance-number>.service-now.com/'
    export SNOW_INSTANCE_UNAME='admin'
    export SNOW_INSTANCE_PWD='<your-instance-password>'
    ```
4) finally run these commands:
  
    ```bash
    pip install browsergym-workarena
    playwright install
    workarena-install
    ```


</details>

<details>
<summary>WebArena on AWS</summary>
TODO
</details>

<details>
<summary>WebArena on Azure</summary>
TODO
</details>





## Launch experiments

Create your agent or import an existing one:
```python
from agentlab.agents.generic_agent.agent_configs import AGENT_4o
```

Run the agent on a benchmark:
```python
study_name, exp_args_list = run_agents_on_benchmark(AGENT_4o, benchmark)
study_dir = make_study_dir(RESULTS_DIR, study_name)
run_experiments(n_jobs, exp_args_list, study_dir)
```

use [main.py](main.py) to launch experiments with a variety
of options. This is like a lazy CLI that is actually more convenient than a CLI.
Just comment and uncomment the lines you need or modify at will (but don't push
to the repo).

<details>

<summary>Debugging</summary>

For debugging, run experiments using `n_jobs=1` and use VSCode debug mode. This
will allow you to stop on breakpoints. To prevent the debugger from stopping
on errors when running multiple experiments directly in VSCode, set
`enable_debug = False` in `ExpArgs` 
</details>





<details>

<summary>Parallel jobs</summary>

Running one agent on one task correspond to one job. When conducting ablation
studies or random searches on hundreds of tasks with multiple seeds, this can
lead to more than 10000 jobs. It is thus crucial to execute them in parallel.
The agent usually wait on the LLM server to return the results or the web server
to update the page. Hence, you can run 10-50 jobs in parallel on a single
computer depending on how much RAM is available.

</details>

## AgentXray
While your experiments are running, you can inspect the results using:

```bash
agentlab-xray
```

<a href="https://github.com/user-attachments/assets/20a91e7b-94ef-423d-9091-743eebb4733d">
  <img src="https://github.com/user-attachments/assets/20a91e7b-94ef-423d-9091-743eebb4733d" width="250" />
</a>

You will be able to select the recent experiments in the directory
`AGENTLAB_EXP_ROOT` and visualize the results in a gradio interface.

In the following order, select:
* The experiment you want to visualize
* The agent if there is more than one
* The task
* And the seed

Once this is selected, you can see the trace of your agent on the given task.
Click on the profiling image to select a step and observe the action taken by the agent.

## Implement a new Agent

Get inspiration from the `MostBasicAgent` in [agentlab/agents/most_basic_agent/most_basic_agent.py](src/agentlab/agents/most_basic_agent/most_basic_agent.py)

Create a new directory in agentlab/agents/ with the name of your agent. 

## Misc

if you want to download HF models more quickly
```
pip install hf-transfer
pip install torch
export HF_HUB_ENABLE_HF_TRANSFER=1
```
