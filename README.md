## Install ui_copilot

:warning: skip this section if you've already installed the `ui_copilot` conda env.

install the package locall with the `-e` flag
From the same directory as this README file run:

```bash
    pip install -e .
```

or `conda env update` from the main ui_copilot directory.

This will ensure that the `PYTHONPATH` will be set correctly.

## Launch experiments

Open and modify `exp_configs.py` and `launch_command.py` to your needs. They are
located in `ui_copilot/experiments/`.

Then launch the experiment with

```bash
    python launch_command.py
```

Avoid pushing these changes to the repo unless they are structural changes.
If you prefer launching with command line, see section [Launch experiments with command line](#launch-experiments-with-command-line).

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

SSH to a server with many cores to get more parallelism. You can use `screen` to
ensure the process keeps running even if you disconnect.

```bash
    screen -S <screen_name>
    python launch_command.py
    # ctrl-a d to detach
    # screen -r <screen_name> to reattach
```

## Visualize results
Open `ui_copilot/experiments/inspect_results.ipynb` in jupyter notebook.

Set your `result_dir` to the right value and run the notebook.



## Launch experiments with command line
Alternatively, you can launch experiments from the command line.

Choose or configure your experiment in `ui_copilot/experiments/exp_configs.py`.
Make sure it is in the EXP_GROUPS global variable.

Then launch the experiment with

```bash
    python src/ui_copilot/experiments/launch_exp.py  \
        --savedir_base=<directory/to/save/experiments> \
        --exp_group_name=<name_of_exp_group> \
        --n_jobs=<joblib_pool_size>
```

For example, this will launch a quick test in the default directory:

```bash
    python src/ui_copilot/experiments/launch_exp.py  \
        --exp_group_name=generic_agent_test \
        --n_jobs=1
```

Some flags are not yet available in the command line. Feel free to add them to
match the interace of main() in `launch_exp.py`.

If you want to test the pipeline of serving OSS LLMs with TGI on Toolkit for evaluation purposes, use `exp_group_name=test_OSS_toolkit` 


## Misc

if you want to download HF models more quickly
```
pip install hf-transfer
pip install torch
export HF_HUB_ENABLE_HF_TRANSFER=1
```
