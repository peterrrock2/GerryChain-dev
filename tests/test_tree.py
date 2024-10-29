import functools

import networkx
import rustworkx
import pytest

from gerrychain import MarkovChain
from gerrychain.constraints import contiguous, within_percent_of_ideal_population
from gerrychain.graph import Graph, add_surcharges
from gerrychain.partition import Partition
from gerrychain.proposals import recom, reversible_recom
from gerrychain.tree import (
    bipartition_tree,
    random_spanning_tree,
    find_balanced_edge_cuts_contraction,
    find_balanced_edge_cuts_memoization,
    recursive_tree_part,
    recursive_seed_part,
    PopulatedGraph,
    uniform_spanning_tree,
    get_max_prime_factor_less_than,
    bipartition_tree_random,
)
from gerrychain.updaters import Tally, cut_edges
from functools import partial
import random

random.seed(2018)


# ADD A TEST FOR RECURSIVE SEED PART WITH N=2


@pytest.fixture
def graph_with_pop(three_by_three_grid: Graph):
    for node in three_by_three_grid.node_indices():
        three_by_three_grid[node]["pop"] = 1
    return add_surcharges(three_by_three_grid)


@pytest.fixture
def partition_with_pop(graph_with_pop: Graph | rustworkx.PyGraph):
    return Partition(
        graph_with_pop,
        {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 1, 7: 1, 8: 1},
        updaters={"pop": Tally("pop"), "cut_edges": cut_edges},
    )


@pytest.fixture
def twelve_by_twelve_with_pop():
    xy_grid = networkx.grid_graph([12, 12])
    nodes = {node: node[1] + 12 * node[0] for node in xy_grid}
    grid = networkx.relabel_nodes(xy_grid, nodes)
    for node in grid:
        grid.nodes[node]["pop"] = 1
    return add_surcharges(Graph.from_networkx(grid))


def test_bipartition_tree_returns_a_subset_of_nodes(
    graph_with_pop: Graph | rustworkx.PyGraph,
):
    ideal_pop = sum(node["pop"] for node in graph_with_pop) / 2
    result = bipartition_tree(graph_with_pop, "pop", ideal_pop, 0.25, 10)
    assert isinstance(result, frozenset)
    assert all(node in {0, 1, 2, 3, 4, 5, 6, 7, 8} for node in result)


def test_bipartition_tree_returns_within_epsilon_of_target_pop(
    graph_with_pop: Graph | rustworkx.PyGraph,
):
    ideal_pop = sum(node["pop"] for node in graph_with_pop) / 2
    epsilon = 0.25
    result = bipartition_tree(graph_with_pop, "pop", ideal_pop, epsilon, 10)

    part_pop = sum(graph_with_pop[node]["pop"] for node in result)
    assert abs(part_pop - ideal_pop) / ideal_pop < epsilon


def test_recursive_tree_part_returns_within_epsilon_of_target_pop(
    twelve_by_twelve_with_pop: Graph | rustworkx.PyGraph,
):
    n_districts = 7  # 144/7 ≈ 20.5 nodes/subgraph (1 person/node)
    ideal_pop = (sum(node["pop"] for node in twelve_by_twelve_with_pop)) / n_districts
    epsilon = 0.05
    result = recursive_tree_part(
        twelve_by_twelve_with_pop,
        range(n_districts),
        ideal_pop,
        "pop",
        epsilon,
    )

    partition = Partition(
        twelve_by_twelve_with_pop, result, updaters={"pop": Tally("pop")}
    )

    assert all(
        abs(part_pop - ideal_pop) / ideal_pop < epsilon
        for part_pop in partition["pop"].values()
    )


def test_recursive_tree_part_returns_within_epsilon_of_target_pop_using_contraction(
    twelve_by_twelve_with_pop,
):
    n_districts = 7  # 144/7 ≈ 20.5 nodes/subgraph (1 person/node)
    ideal_pop = (sum(node["pop"] for node in twelve_by_twelve_with_pop)) / n_districts
    epsilon = 0.05
    result = recursive_tree_part(
        twelve_by_twelve_with_pop,
        range(n_districts),
        ideal_pop,
        "pop",
        epsilon,
        method=partial(
            bipartition_tree,
            max_attempts=10000,
            balance_edge_fn=find_balanced_edge_cuts_contraction,
        ),
    )
    partition = Partition(
        twelve_by_twelve_with_pop, result, updaters={"pop": Tally("pop")}
    )
    assert all(
        abs(part_pop - ideal_pop) / ideal_pop < epsilon
        for part_pop in partition["pop"].values()
    )


def test_recursive_seed_part_returns_within_epsilon_of_target_pop(
    twelve_by_twelve_with_pop,
):
    n_districts = 7  # 144/7 ≈ 20.5 nodes/subgraph (1 person/node)
    ideal_pop = (sum(node["pop"] for node in twelve_by_twelve_with_pop)) / n_districts
    epsilon = 0.1
    result = recursive_seed_part(
        twelve_by_twelve_with_pop,
        range(n_districts),
        ideal_pop,
        "pop",
        epsilon,
        n=5,
        ceil=None,
    )
    partition = Partition(
        twelve_by_twelve_with_pop, result, updaters={"pop": Tally("pop")}
    )
    assert all(
        abs(part_pop - ideal_pop) / ideal_pop < epsilon
        for part_pop in partition["pop"].values()
    )


def test_recursive_seed_part_returns_within_epsilon_of_target_pop_using_contraction(
    twelve_by_twelve_with_pop,
):
    n_districts = 7  # 144/7 ≈ 20.5 nodes/subgraph (1 person/node)
    ideal_pop = (sum(node["pop"] for node in twelve_by_twelve_with_pop)) / n_districts
    epsilon = 0.1
    result = recursive_seed_part(
        twelve_by_twelve_with_pop,
        range(n_districts),
        ideal_pop,
        "pop",
        epsilon,
        n=5,
        ceil=None,
        method=partial(
            bipartition_tree,
            max_attempts=10000,
            balance_edge_fn=find_balanced_edge_cuts_contraction,
        ),
    )
    partition = Partition(
        twelve_by_twelve_with_pop, result, updaters={"pop": Tally("pop")}
    )
    assert all(
        abs(part_pop - ideal_pop) / ideal_pop < epsilon
        for part_pop in partition["pop"].values()
    )


def test_recursive_seed_part_uses_method(twelve_by_twelve_with_pop):
    calls = 0

    def dummy_method(graph, pop_col, pop_target, epsilon, node_repeats, one_sided_cut):
        nonlocal calls
        calls += 1
        return bipartition_tree(
            graph,
            pop_col=pop_col,
            pop_target=pop_target,
            epsilon=epsilon,
            node_repeats=node_repeats,
            max_attempts=10000,
            one_sided_cut=one_sided_cut,
        )

    n_districts = 7  # 144/7 ≈ 20.5 nodes/subgraph (1 person/node)
    ideal_pop = (sum(node["pop"] for node in twelve_by_twelve_with_pop)) / n_districts
    epsilon = 0.1
    result = recursive_seed_part(
        twelve_by_twelve_with_pop,
        range(n_districts),
        ideal_pop,
        "pop",
        epsilon,
        n=5,
        ceil=None,
        method=dummy_method,
    )
    # Called at least once for each district besides the last one
    # (note that current implementation of recursive_seed_part calls method
    # EXACTLY once for each district besides the last one, but that is an
    # implementation detail)
    assert calls >= n_districts - 1


def test_recursive_seed_part_with_n_unspecified_within_epsilon(
    twelve_by_twelve_with_pop,
):
    n_districts = 6  # This should set n=3
    ideal_pop = (sum(node["pop"] for node in twelve_by_twelve_with_pop)) / n_districts
    epsilon = 0.05
    result = recursive_seed_part(
        twelve_by_twelve_with_pop,
        range(n_districts),
        ideal_pop,
        "pop",
        epsilon,
        ceil=None,
    )
    partition = Partition(
        twelve_by_twelve_with_pop, result, updaters={"pop": Tally("pop")}
    )
    assert all(
        abs(part_pop - ideal_pop) / ideal_pop < epsilon
        for part_pop in partition["pop"].values()
    )


def test_random_spanning_tree_returns_tree_with_pop_attribute(graph_with_pop):
    tree = random_spanning_tree(graph_with_pop)
    nx_graph = Graph.to_networkx(tree)
    assert networkx.is_tree(nx_graph)


def test_uniform_spanning_tree_returns_tree_with_pop_attribute(graph_with_pop):
    tree = uniform_spanning_tree(graph_with_pop)
    nx_graph = Graph.to_networkx(tree)
    assert networkx.is_tree(nx_graph)


def test_bipartition_tree_returns_a_tree(graph_with_pop: Graph | rustworkx.PyGraph):
    ideal_pop = (
        sum(
            graph_with_pop.get_node_data(idx)[1]["pop"]
            for idx in graph_with_pop.node_indices()
        )
        / 2
    )
    tree = Graph.from_networkx(
        networkx.Graph([(0, 1), (1, 2), (1, 4), (3, 4), (4, 5), (3, 6), (6, 7), (6, 8)])
    )
    for idx in tree.node_indices():
        tree.get_node_data(idx)[1]["pop"] = graph_with_pop.get_node_data(idx)[1]["pop"]

    result = bipartition_tree(
        graph_with_pop, "pop", ideal_pop, 0.25, 10, tree, lambda x: 4
    )

    result_graph = rustworkx.PyGraph()
    result_graph.add_nodes_from([graph_with_pop.get_node_data(idx) for idx in result])

    subgraph = graph_with_pop.subgraph(list(result))
    result_graph.add_edges_from(
        [
            (
                e[0],
                e[1],
                graph_with_pop.get_edge_data(
                    subgraph.get_node_data(e[0])[0], subgraph.get_node_data(e[1])[0]
                ),
            )
            for e in subgraph.edge_list()
        ]
    )

    nx_graph = Graph.to_networkx(result_graph)

    assert networkx.is_tree(Graph.to_networkx(result_graph))

    remaining_nodes = set(graph_with_pop.node_indices()) - set(result)
    remaining_edges = [
        (graph_with_pop.get_node_data(u)[0], graph_with_pop.get_node_data(v)[0], data)
        for u, v, data in graph_with_pop.weighted_edge_list()
        if u in remaining_nodes and v in remaining_nodes
    ]

    remaining_graph = rustworkx.PyGraph()
    remaining_graph.add_nodes_from(
        [graph_with_pop.get_node_data(idx) for idx in remaining_nodes]
    )

    old_to_new_nodes = {
        remaining_graph.get_node_data(idx)[0]: idx
        for idx in remaining_graph.node_indices()
    }

    remaining_graph.add_edges_from(
        [
            (old_to_new_nodes[u], old_to_new_nodes[v], data)
            for u, v, data in remaining_edges
        ]
    )

    assert networkx.is_tree(Graph.to_networkx(remaining_graph))


def test_recom_works_as_a_proposal(partition_with_pop):
    graph = partition_with_pop.graph
    # print(graph)
    # for node in graph:
    #     print(node)
    ideal_pop = sum(node["pop"] for node in graph) / 2
    proposal = functools.partial(
        recom, pop_col="pop", pop_target=ideal_pop, epsilon=0.25, node_repeats=5
    )
    constraints = [contiguous]

    chain = MarkovChain(proposal, constraints, lambda x: True, partition_with_pop, 100)

    for state in chain:
        assert contiguous(state)


def test_reversible_recom_works_as_a_proposal(partition_with_pop):
    random.seed(2018)
    graph = partition_with_pop.graph
    ideal_pop = sum(node["pop"] for node in graph) / 2
    proposal = functools.partial(
        reversible_recom, pop_col="pop", pop_target=ideal_pop, epsilon=0.10, M=1
    )
    constraints = [within_percent_of_ideal_population(partition_with_pop, 0.25, "pop")]

    chain = MarkovChain(proposal, constraints, lambda x: True, partition_with_pop, 100)

    for state in chain:
        assert contiguous(state)


def test_find_balanced_cuts_contraction():
    tree = Graph.from_networkx(
        networkx.Graph([(0, 1), (1, 2), (1, 4), (3, 4), (4, 5), (3, 6), (6, 7), (6, 8)])
    )

    #  0 - 1 - 2
    #     ||
    #  4= 3 - 5
    # ||
    # 6- 7
    # |
    # 8

    populated_tree = PopulatedGraph(
        tree, {node: 1 for node in tree.node_indices()}, len(tree) / 2, 0.5
    )
    cuts = find_balanced_edge_cuts_contraction(populated_tree)
    edges = set(tuple(sorted(cut.edge)) for cut in cuts)
    assert edges == {(1, 3), (3, 4), (4, 6)}


def test_no_balanced_cuts_contraction_when_one_side_okay():
    tree = Graph.from_networkx(networkx.Graph([(0, 1), (1, 2), (2, 3), (3, 4)]))

    populations = {0: 4, 1: 4, 2: 3, 3: 3, 4: 3}

    populated_tree = PopulatedGraph(
        graph=tree, populations=populations, ideal_pop=10, epsilon=0.1
    )

    cuts = find_balanced_edge_cuts_contraction(populated_tree, one_sided_cut=False)
    assert cuts == []


def test_find_balanced_cuts_memo():
    tree = Graph.from_networkx(
        networkx.Graph([(0, 1), (1, 2), (1, 3), (3, 4), (3, 5), (4, 6), (6, 7), (6, 8)])
    )

    #  0 - 1 - 2
    #     ||
    #  4= 3 - 5
    # ||
    # 6- 7
    # |
    # 8

    populated_tree = PopulatedGraph(
        tree, {node: 1 for node in tree.node_indices()}, len(tree) / 2, 0.5
    )

    # THESE CUTS ARE IN TERMS OF HOWEVER RUSTWORKX INDEXES THE NODES
    cuts = find_balanced_edge_cuts_memoization(populated_tree)
    edges = set(tuple(sorted(cut.edge)) for cut in cuts)
    assert edges == {(1, 3), (3, 4), (4, 6)}


def test_no_balanced_cuts_memo_when_one_side_okay():
    tree = Graph.from_networkx(networkx.Graph([(0, 1), (1, 2), (2, 3), (3, 4)]))

    populations = {0: 4, 1: 4, 2: 3, 3: 3, 4: 3}

    populated_tree = PopulatedGraph(
        graph=tree, populations=populations, ideal_pop=10, epsilon=0.1
    )

    cuts = find_balanced_edge_cuts_memoization(populated_tree)
    assert cuts == []


def test_prime_bound():
    assert (
        get_max_prime_factor_less_than(2024, 20) == 11
        and get_max_prime_factor_less_than(2024, 1) == None
        and get_max_prime_factor_less_than(2024, 2000) == 23
        and get_max_prime_factor_less_than(2024, -1) == None
    )


def test_bipartition_tree_random_returns_a_subset_of_nodes(graph_with_pop):
    ideal_pop = sum(node["pop"] for node in graph_with_pop) / 2
    result = bipartition_tree_random(graph_with_pop, "pop", ideal_pop, 0.25, 10)
    assert isinstance(result, frozenset)
    graph_nodes = set(graph_with_pop.node_indices())
    assert all(node in graph_nodes for node in result)


def test_bipartition_tree_random_returns_within_epsilon_of_target_pop(graph_with_pop):
    ideal_pop = sum(node["pop"] for node in graph_with_pop) / 2
    epsilon = 0.25
    result = bipartition_tree_random(graph_with_pop, "pop", ideal_pop, epsilon, 10)

    part_pop = sum(graph_with_pop.get_node_data(node)[1]["pop"] for node in result)
    assert abs(part_pop - ideal_pop) / ideal_pop < epsilon
