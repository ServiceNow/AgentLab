## Install agentlab

:warning: skip this section if you've already installed the `agentlab` conda env.

install the package locall with the `-e` flag
From the same directory as this README file run:

```bash
    pip install -e .
```

or `conda env update` from the main agentlab directory.

This will ensure that the `PYTHONPATH` will be set correctly.

## Launch experiments

We provide default settings to run our agents on a few benchmarks. Those can be ran from `launch_exp.py`,
with the following flags:

```bash
  -h, --help            show this help message and exit
  --exp_root EXP_ROOT   folder where experiments will be saved
  --n_jobs N_JOBS       number of parallel jobs
  --exp_group_name EXP_GROUP_NAME
                        Name of the experiment group to launch as defined in exp_configs.py
  --benchmark {miniwob,workarena.l1,workarena.l2,workarena.l3}
                        Benchmark to launch
  --model_name {gpt-3.5,gpt-4o,gpt-4o-vision,cheat,custom}
                        Model to launch
  --relaunch_mode {None,incomplete_only,all_errors,server_errors}
                        Find all incomplete experiments and relaunch them.
```

`exp_group_name` is the name of the experiment group to launch as defined in `exp_configs.py`. This will override the `benchmark` and `model_name` flags.

As an example, to launch our agent with GPT-4o on the miniwob benchmark, with maximum jobs, run:

```bash
    python src/agentlab/experiments/launch_exp.py  \
        --benchmark=miniwob \
        --model_name=gpt-4o \
        --n_jobs=-1
```

The `cheat` model uses regular expressions to solve miniwob click tasks. It is useful for debugging.

In `exp_configs.py`, you can modify `FLAGS_CUSTOM` and `AGENT_CUSTOM` to test out your own flags and models, and then launch them with `--model_name=custom`.

### Custom experiments

Alternatively, you can customize your experiments by modifying `exp_configs.py` and `launch_command.py`. They are located in `agentlab/experiments/`.

Then launch the experiment with

```bash
    python src/agentlab/experiments/launch_command.py
```


### Debugging jobs

If you launch via VSCode in debug mode, debugging will be enabled and errors will be raised
instead of being logged, unless you set `enable_debug = False` in `ExpArgs`. This
will bring a breakpoint on the error.

To make sure you get a breakpoint at the origin of the error, use the flag
`use_threads_instead_of_processes=True` in `main()` from `launch_exp.py` (or set `n_jobs=1`).


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
