# AgentLab

AgentLab is a framework for developing and evaluating web agents on a variety of
benchmarks supported by [BrowserGym](https://github.com/ServiceNow/BrowserGym).
This includes:
* WebArena
* WorkArena.L1, L2, L3
* VisualWebArena (coming soon)
* MiniWoB

The framework enable the desing of rich hyperparameter spaces and the launch of
parallel experiments using ablation studies or random searches. It also provides
agent_xray, a visualization tool to inspect the results of the experiments using
a custom gradio interface

<a href="https://github.com/user-attachments/assets/20a91e7b-94ef-423d-9091-743eebb4733d">
  <img src="https://github.com/user-attachments/assets/20a91e7b-94ef-423d-9091-743eebb4733d" width="200" />
</a>

## Install agentlab

This repo is intended for developing new agents, hence we clone and install using the `-e` flag.

```bash
git clone git@github.com:ServiceNow/AgentLab.git
pip install -e .
```

## Set Environment Variables

```bash
export AGENTLAB_RESULTS_DIR=<root directory of experiment results>  # defaults to $HOME/agentlab_results
export OPENAI_API_KEY=<your openai api key> # if openai models are used
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
<summary>WebArena on AWS</summary>
TODO
</details>

<details>
<summary>WebArena on Azure</summary>
TODO
</details>


<details>

<summary>WorkArena</summary>

```bash
export SNOW_INSTANCE_URL="https://<your-instance-number>.service-now.com/"
export SNOW_INSTANCE_UNAME="admin"
export SNOW_INSTANCE_PWD=<your-instance-password>
```

</details>


## Launch experiments

Experiments can be ran from `launch_exp.py`, with the following options:

```bash
  -h, --help            show this help message and exit
  --exp_root EXP_ROOT   folder where experiments will be saved
  --n_jobs N_JOBS       number of parallel jobs
  --exp_config EXP_CONFIG
                        Python path to the experiment function to launch
  --benchmark {miniwob,workarena.l1,workarena.l2,workarena.l3}
                        Benchmark to launch
  --agent_config AGENT_CONFIG
                        Python path to the agent config
  --relaunch_mode {None,incomplete_only,all_errors,server_errors}
                        Find all incomplete experiments and relaunch them.
  --extra_kwargs EXTRA_KWARGS
                        Extra arguments to pass to the experiment group.
```

`exp_config` is a python path to the experiment function to launch as defined in `exp_configs.py`. This function must return a list of `ExpArgs` objects, that correspond to running one agent on a task.

`agent_config` is a python path to an agent config.

Our experiment and agent configs are available at `agentlab.agents.generic_agent`

As an example, to launch our agent with GPT-4o on the miniwob benchmark, with maximum jobs, run:

```bash
    python src/agentlab/experiments/launch_exp.py  \
        --exp_config=agentlab.agents.generic_agent.run_agent_on_benchmark \
        --agent_config=agentlab.agents.generic_agent.AGENT_4o \
        --benchmark=miniwob \
        --n_jobs=-1
```

Our configs are available in the [agent_config folder](src/agentlab/agents/generic_agent/agent_configs.py). We provide configs for our agent with the following models:
- `AGENT_3_5`: GPT-3.5
- `AGENT_4o`: GPT-4o
- `AGENT_4o_vision`: GPT-4o with vision
- `AGENT_8B`: Llama3-8B
- `AGENT_70B`: Llama3-70B

We additionnaly provide an `AGENT_CUSTOM` config that can be used to try out flags.

### Custom experiments

Alternatively, you can customize your experiments by modifying `exp_configs.py` and `launch_command.py`. They are located respectively in `agentlab/agents/generic_agent` and `agentlab/experiments/`.

Then launch the experiment with

```bash
    python src/agentlab/experiments/launch_command.py
```


### Debugging

If you launch via VSCode in debug mode, debugging will be enabled and errors will be raised
instead of being logged, unless you set `enable_debug = False` in `ExpArgs`. This
will bring a breakpoint on the error.

To make sure you get a breakpoint at the origin of the error, we recommend to set `n_jobs=1` in `main()` from `launch_exp.py`.


### `joblib`'s parallel jobs
Jobs are launched in parallel using joblib. This will launch multiple processes
on a single computer. The choice is based on the fact that, in general, we are not CPU
bounded. If it becomes the bottleneck we can launch using multiple servers.

SSH to a server with many cores to get more parallelism. You can use `screen` (or `tmux`) to
ensure the process keeps running even if you disconnect.

```bash
    screen -S <screen_name>
    python launch_command.py
    # ctrl-a d to detach
    # screen -r <screen_name> to reattach
```

## Visualize results
Open `agentlab/experiments/inspect_results.ipynb` in jupyter notebook.

Set your `result_dir` to the right value and run the notebook.

## Misc

if you want to download HF models more quickly
```
pip install hf-transfer
pip install torch
export HF_HUB_ENABLE_HF_TRANSFER=1
```