"""Validate experiment dependencies before launching workers or model servers."""

from heapq import heappop, heappush


class TaskGraph:
    """A validated snapshot of experiments with a stable topological order.

    IDs must already have been assigned (``ExpArgs.make_id()`` does this without
    preparing an experiment directory). Original experiment objects are retained;
    neither the caller's list nor its dependency collections are modified. Among
    runnable experiments, the earliest input position is chosen first.
    """

    def __init__(self, experiments):
        self.experiments = tuple(experiments)
        self.by_id = {}
        self.dependencies = {}
        for index, experiment in enumerate(self.experiments):
            exp_id = experiment.exp_id
            if not isinstance(exp_id, str) or not exp_id:
                raise ValueError(f"Invalid experiment ID at position {index}: {exp_id!r}")
            if exp_id in self.by_id:
                raise ValueError(f"Duplicate experiment ID: {exp_id!r}")
            self.by_id[exp_id] = experiment

        for experiment in self.experiments:
            exp_id = experiment.exp_id
            dependencies = experiment.depends_on
            if not isinstance(dependencies, (tuple, list)):
                raise ValueError(f"depends_on for {exp_id!r} must be a list or tuple of IDs")
            unique_dependencies = []
            seen = set()
            for dependency in dependencies:
                if not isinstance(dependency, str) or not dependency:
                    raise ValueError(f"Invalid dependency ID for {exp_id!r}: {dependency!r}")
                if dependency not in self.by_id:
                    raise ValueError(f"Experiment {exp_id!r} depends on missing ID {dependency!r}")
                if dependency not in seen:
                    seen.add(dependency)
                    unique_dependencies.append(dependency)
            self.dependencies[exp_id] = tuple(unique_dependencies)

        positions = {exp_id: index for index, exp_id in enumerate(self.by_id)}
        remaining = {exp_id: len(deps) for exp_id, deps in self.dependencies.items()}
        dependents = {exp_id: [] for exp_id in self.by_id}
        for exp_id, dependencies in self.dependencies.items():
            for dependency in dependencies:
                dependents[dependency].append(exp_id)

        ready = []
        for exp_id, count in remaining.items():
            if count == 0:
                heappush(ready, positions[exp_id])
        ordered = []
        while ready:
            experiment = self.experiments[heappop(ready)]
            ordered.append(experiment)
            for dependent in dependents[experiment.exp_id]:
                remaining[dependent] -= 1
                if remaining[dependent] == 0:
                    heappush(ready, positions[dependent])

        if len(ordered) != len(self.experiments):
            unresolved = [exp_id for exp_id, count in remaining.items() if count]
            raise ValueError(f"Dependency cycle involving experiments: {', '.join(unresolved)}")
        self.ordered_experiments = tuple(ordered)
        self.has_dependencies = any(self.dependencies.values())
