from agentlab.analyze import inspect_results

result_df = inspect_results.load_result_df("./agentlab_results")

print(result_df.head())
