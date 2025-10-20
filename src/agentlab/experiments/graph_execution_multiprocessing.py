"""
Multiprocessing-based task graph execution for AgentLab experiments.

This module provides a simpler alternative to Ray for parallel execution,
using Python's standard multiprocessing library. It executes tasks in layers
based on their dependencies (topological sort).
"""

import logging
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import bgym

from agentlab.experiments.exp_utils import _episode_timeout, run_exp

logger = logging.getLogger(__name__)


def build_dependency_graph(exp_args_list: list[bgym.ExpArgs]) -> dict[str, list[str]]:
    """
    Build a dependency graph from experiment arguments.
    
    Args:
        exp_args_list: List of experiment arguments
        
    Returns:
        Dictionary mapping exp_id to list of dependency exp_ids
    """
    dependency_graph = {}
    for exp_args in exp_args_list:
        dependency_graph[exp_args.exp_id] = list(exp_args.depends_on)
    return dependency_graph


def topological_sort_layers(exp_args_list: list[bgym.ExpArgs]) -> list[list[bgym.ExpArgs]]:
    """
    Perform topological sort and group tasks into layers.
    
    Each layer contains tasks that can be executed in parallel (no dependencies
    on each other, only on previous layers).
    
    Args:
        exp_args_list: List of experiment arguments
        
    Returns:
        List of layers, where each layer is a list of ExpArgs that can run in parallel
        
    Raises:
        ValueError: If circular dependency is detected
    """
    # Build mappings
    exp_args_map = {exp_args.exp_id: exp_args for exp_args in exp_args_list}
    dependency_graph = build_dependency_graph(exp_args_list)
    
    # Calculate in-degree (number of dependencies) for each task
    in_degree = {exp_id: len(deps) for exp_id, deps in dependency_graph.items()}
    
    # Find all tasks with no dependencies
    ready_tasks = deque([exp_id for exp_id, degree in in_degree.items() if degree == 0])
    
    layers = []
    processed_count = 0
    
    while ready_tasks:
        # Current layer: all tasks that are ready to run
        current_layer_ids = list(ready_tasks)
        current_layer = [exp_args_map[exp_id] for exp_id in current_layer_ids]
        layers.append(current_layer)
        
        # Clear the ready queue
        ready_tasks.clear()
        
        # Process each task in the current layer
        for exp_id in current_layer_ids:
            processed_count += 1
            
            # For each task that depends on this one, decrease its in-degree
            for other_exp_id, deps in dependency_graph.items():
                if exp_id in deps:
                    in_degree[other_exp_id] -= 1
                    # If all dependencies are satisfied, add to ready queue
                    if in_degree[other_exp_id] == 0:
                        ready_tasks.append(other_exp_id)
    
    # Check for circular dependencies
    if processed_count != len(exp_args_list):
        raise ValueError(
            f"Circular dependency detected! Processed {processed_count} tasks "
            f"out of {len(exp_args_list)} total tasks."
        )
    
    return layers


def execute_task_graph_multiprocessing(
    exp_args_list: list[bgym.ExpArgs],
    n_jobs: int,
    avg_step_timeout: int = 60,
) -> dict[str, Any]:
    """
    Execute a task graph while respecting dependencies using multiprocessing.
    
    This function organizes tasks into layers based on their dependencies and
    executes each layer in parallel using ProcessPoolExecutor.
    
    Args:
        exp_args_list: List of experiment arguments with dependency information
        n_jobs: Number of parallel processes to use
        avg_step_timeout: Average timeout per step (for logging warnings)
        
    Returns:
        Dictionary mapping exp_id to execution result (or exception)
        
    Note:
        Unlike Ray backend, this does not actively cancel timeout tasks.
        It only logs warnings when tasks exceed expected duration.
    """
    if not exp_args_list:
        logger.warning("No experiments to run.")
        return {}
    
    logger.info(f"Organizing {len(exp_args_list)} tasks into execution layers...")
    
    # Organize tasks into layers
    layers = topological_sort_layers(exp_args_list)
    
    logger.info(f"Organized tasks into {len(layers)} layers:")
    for i, layer in enumerate(layers):
        logger.info(f"  Layer {i}: {len(layer)} tasks")
    
    results = {}
    
    # Execute each layer sequentially, but tasks within each layer in parallel
    for layer_idx, layer in enumerate(layers):
        logger.info(f"Executing layer {layer_idx + 1}/{len(layers)} with {len(layer)} tasks...")
        
        # Calculate timeout for this layer
        layer_timeouts = [_episode_timeout(exp_args, avg_step_timeout) for exp_args in layer]
        max_timeout = max(layer_timeouts) if layer_timeouts else 3600
        
        # Execute tasks in this layer in parallel
        layer_results = _execute_layer(layer, n_jobs, max_timeout, avg_step_timeout)
        results.update(layer_results)
        
        logger.info(f"Completed layer {layer_idx + 1}/{len(layers)}")
    
    logger.info("All layers completed.")
    return results


def _execute_layer(
    layer: list[bgym.ExpArgs],
    n_jobs: int,
    max_timeout: float,
    avg_step_timeout: int,
) -> dict[str, Any]:
    """
    Execute all tasks in a single layer in parallel.
    
    Args:
        layer: List of ExpArgs for this layer
        n_jobs: Number of parallel workers
        max_timeout: Maximum expected timeout for any task in this layer
        avg_step_timeout: Average timeout per step
        
    Returns:
        Dictionary mapping exp_id to result
    """
    results = {}
    
    # Use ProcessPoolExecutor for parallel execution
    with ProcessPoolExecutor(max_workers=min(n_jobs, len(layer))) as executor:
        # Submit all tasks
        future_to_exp_id = {}
        for exp_args in layer:
            future = executor.submit(run_exp, exp_args, avg_step_timeout=avg_step_timeout)
            future_to_exp_id[future] = exp_args.exp_id
        
        # Collect results as they complete
        for future in as_completed(future_to_exp_id):
            exp_id = future_to_exp_id[future]
            try:
                result = future.result(timeout=max_timeout + 60)  # Add buffer to timeout
                results[exp_id] = result
                logger.debug(f"Task {exp_id} completed successfully")
            except Exception as e:
                logger.error(f"Task {exp_id} failed with error: {e}")
                results[exp_id] = e
    
    return results
