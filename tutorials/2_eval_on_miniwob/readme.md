### Launch experiment
* See the `launch_experiments.ipynb` to setup miniwob benchmark.
* Export MINIWOB_URL in your envronment variables.
* Run the following command from **AgentLab** directory `uv run tutorials/2_eval_on_miniwob/experiment.py`.
* This should launch your agent on 4 miniwob tasks in paralel and save results to `$HOME/agentlab_results`.


### Visualize experiments in XRay
* run `agentlab-xray`
* select experiment directory
* Navigate around XRay to visualize the agent's behavior

### (optional) Load your results in a notebook
Optionnaly load the results in `inspect_results.ipynb`.
This will help understanding the content of results

### Change the experiment
go back to `experiment.py` and uncomment the line 
```python
benchmark = DEFAULT_BENCHMARKS["miniwob"]()  # 125 tasks
benchmark = benchmark.subset_from_glob(column="task_name", glob="*enter*")
```
This will change the benchmark from `"miniwob_tiny_test"` to all the `"*enter*"` tasks in the full miniwob.

Run the experiment again. This will launch on 7 tasks with 5 seeds each.

Finally refresh XRay to be able to load the new experiment.