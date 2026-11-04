"""The graph contract is independent of browsers, agents, and Ray."""

from types import SimpleNamespace

import pytest

from agentlab.experiments.task_graph import TaskGraph


def experiment(exp_id, *dependencies):
    return SimpleNamespace(exp_id=exp_id, depends_on=dependencies)


def test_empty_graph():
    graph = TaskGraph([])
    assert graph.experiments == ()
    assert graph.ordered_experiments == ()
    assert not graph.has_dependencies


def test_independent_experiments_preserve_input_order():
    experiments = [experiment("z"), experiment("a"), experiment("middle")]
    graph = TaskGraph(experiments)
    assert graph.experiments == tuple(experiments)
    assert graph.ordered_experiments == tuple(experiments)
    assert graph.by_id["z"] is experiments[0]


def test_stable_topological_order_for_diamond():
    experiments = [
        experiment("finish", "left", "right"),
        experiment("right", "start"),
        experiment("unrelated"),
        experiment("left", "start"),
        experiment("start"),
    ]
    graph = TaskGraph(experiments)
    assert [exp.exp_id for exp in graph.ordered_experiments] == [
        "unrelated",
        "start",
        "right",
        "left",
        "finish",
    ]
    assert graph.has_dependencies
    assert graph.dependencies["finish"] == ("left", "right")


def test_graph_does_not_reorder_caller_list_or_mutate_dependencies():
    child, parent = experiment("child", "parent", "parent"), experiment("parent")
    experiments = [child, parent]
    graph = TaskGraph(experiments)
    assert experiments == [child, parent]
    assert child.depends_on == ("parent", "parent")
    assert graph.dependencies["child"] == ("parent",)
    assert graph.ordered_experiments == (parent, child)


def test_long_chain_does_not_recurse():
    experiments = [experiment(str(i), str(i + 1)) for i in range(2500)]
    experiments.append(experiment("2500"))
    graph = TaskGraph(experiments)
    assert [exp.exp_id for exp in graph.ordered_experiments] == [
        str(i) for i in range(2500, -1, -1)
    ]


@pytest.mark.parametrize("bad_id", [None, "", [], {}, 7])
def test_invalid_experiment_id_is_actionable(bad_id):
    with pytest.raises(ValueError, match="experiment ID"):
        TaskGraph([experiment(bad_id)])


def test_duplicate_id_is_not_silently_overwritten():
    with pytest.raises(ValueError, match="Duplicate experiment ID.*same"):
        TaskGraph([experiment("same"), experiment("same")])


def test_repeated_experiment_object_is_also_a_duplicate():
    exp = experiment("same")
    with pytest.raises(ValueError, match="Duplicate experiment ID"):
        TaskGraph([exp, exp])


def test_unknown_dependency_names_both_experiments():
    with pytest.raises(ValueError, match="child.*missing"):
        TaskGraph([experiment("child", "missing")])


@pytest.mark.parametrize("dependencies", [None, "parent", 17, {"parent": True}])
def test_dependency_collection_must_be_a_list_or_tuple(dependencies):
    exp = SimpleNamespace(exp_id="child", depends_on=dependencies)
    with pytest.raises(ValueError, match="depends_on.*child"):
        TaskGraph([exp, experiment("parent")])


@pytest.mark.parametrize("dependency", [None, "", [], 7])
def test_dependency_ids_are_validated_before_lookup(dependency):
    with pytest.raises(ValueError, match="dependency ID.*child"):
        TaskGraph([experiment("child", dependency)])


def test_self_dependency_reports_cycle():
    with pytest.raises(ValueError, match="cycle.*self"):
        TaskGraph([experiment("self", "self")])


def test_cycle_in_disconnected_component_is_rejected():
    with pytest.raises(ValueError, match="cycle.*a.*b"):
        TaskGraph([experiment("valid"), experiment("a", "b"), experiment("b", "a")])


def test_cycle_with_dependent_leaf_reports_unresolved_ids():
    with pytest.raises(ValueError, match="cycle") as error:
        TaskGraph([experiment("leaf", "a"), experiment("a", "b"), experiment("b", "a")])
    assert "a" in str(error.value)
    assert "b" in str(error.value)


def test_snapshot_is_not_changed_by_mutating_input_collection():
    experiments = [experiment("a")]
    graph = TaskGraph(experiments)
    experiments.clear()
    assert len(graph.experiments) == 1
