import pandas
import pytest

from gerrychain import Partition, Graph
from gerrychain.graph import add_surcharges
from frozendict import frozendict


@pytest.fixture
def seven_by_eight_grid_single_join():
    """
    Not quite a full grid because I needed to split the halves to
    check the attenuation.

     0  1  2  3  4  5  6  7
     8  9 10 11 12 13 14 15
              "  "
    16 17 18~19 20~21 22 23
           "  |  |  "
    24 25~26-27=28-29~30 31
           "  |  |  "
    32 33 34~35 36~37 38 39
              "  "
    40 41 42 43 44 45 46 47
    48 49 50 51 52 53 54 55
    """

    graph = Graph()
    graph.add_nodes_from(list(range(42)))

    edges = []
    cols = 8
    rows = 7
    for i in range(rows):
        for j in range(cols):
            node = i * cols + j
            if j < cols - 1:
                edges.append((node, node + 1))
            if i < rows - 1:
                edges.append((node, node + cols))

    edges.remove((3, 4))
    edges.remove((11, 12))
    edges.remove((19, 20))
    edges.remove((35, 36))
    edges.remove((43, 44))
    edges.remove((51, 52))

    graph.add_edges_from(edges)

    for node in graph.nodes:
        graph.nodes[node]["TOTPOP"] = 1
        graph.nodes[node]["region"] = int(node % 8 < 4)

    for node in graph.nodes:
        graph.nodes[node]["district"] = 1 if node <= 27 else 2

    return graph


@pytest.fixture
def four_by_eight_grid():
    """
     0  1~ 2- 3= 4- 5~ 6  7
           "  |  |  "
     8  9~10-11=12-13~14 15
           "  |  |  "
    16 17~18-19=20-21~22 23
           "  |  |  "
    24 25~26-27=28-29~30 31
    """

    graph = Graph()
    graph.add_nodes_from(list(range(42)))

    edges = []
    cols = 8
    rows = 4
    for i in range(rows):
        for j in range(cols):
            node = i * cols + j
            if j < cols - 1:
                edges.append((node, node + 1))
            if i < rows - 1:
                edges.append((node, node + cols))

    graph.add_edges_from(edges)

    for node in graph.nodes:
        graph.nodes[node]["TOTPOP"] = 1
        graph.nodes[node]["region"] = int(node % 8 < 4)

    for node in graph.nodes:
        graph.nodes[node]["district"] = 1 if node <= cols * rows // 2 else 2

    return graph


@pytest.fixture
def three_by_nine_grid_tripartition():
    """
     0  1  2- 3  4  5+ 6  7  8
     9 10 11-12 13 14+15 16 17
    18 19 20-21 22 23+24 25 26
    """

    graph = Graph()
    graph.add_nodes_from(list(range(42)))

    edges = []
    cols = 9
    rows = 3
    for i in range(rows):
        for j in range(cols):
            node = i * cols + j
            if j < cols - 1:
                edges.append((node, node + 1))
            if i < rows - 1:
                edges.append((node, node + cols))

    graph.add_edges_from(edges)

    for node in graph.nodes:
        graph.nodes[node]["TOTPOP"] = 1
        if node % 9 < 3:
            graph.nodes[node]["region"] = 0
            graph.nodes[node]["coi_1"] = 0
            graph.nodes[node]["coi_2"] = None
            graph.nodes[node]["coi_3"] = 0
        elif node % 9 < 6:
            graph.nodes[node]["region"] = 1
            graph.nodes[node]["coi_1"] = None
            graph.nodes[node]["coi_2"] = None
            graph.nodes[node]["coi_3"] = None
        else:
            graph.nodes[node]["region"] = 2
            graph.nodes[node]["coi_1"] = None
            graph.nodes[node]["coi_2"] = 0
            graph.nodes[node]["coi_3"] = 1

    for node in graph.nodes:
        graph.nodes[node]["district"] = 1 if node <= cols * rows // 2 else 2

    return graph


@pytest.fixture
def two_by_ten_grid_subregion_distinct():
    """
     0  1+ 2  3  4- 5+ 6  7  8  9
    10 11+12 13 14-15+16 17 18 19
    """

    graph = Graph()
    graph.add_nodes_from(list(range(42)))

    edges = []
    cols = 10
    rows = 2
    for i in range(rows):
        for j in range(cols):
            node = i * cols + j
            if j < cols - 1:
                edges.append((node, node + 1))
            if i < rows - 1:
                edges.append((node, node + cols))

    graph.add_edges_from(edges)

    for node in graph.nodes:
        graph.nodes[node]["TOTPOP"] = 1
        if node % 10 < 2:
            graph.nodes[node]["region"] = 0
            graph.nodes[node]["subregion"] = 0
        elif node % 10 < 5:
            graph.nodes[node]["region"] = 0
            graph.nodes[node]["subregion"] = 1
        elif node % 10 == 5:
            graph.nodes[node]["region"] = 1
            graph.nodes[node]["subregion"] = 1
        else:
            graph.nodes[node]["region"] = 1
            graph.nodes[node]["subregion"] = 2

    for node in graph.nodes:
        graph.nodes[node]["district"] = 1 if node <= cols * rows // 2 else 2

    return graph


@pytest.fixture
def two_by_ten_grid_subregion_overlap():
    """
     0  1+ 2  3  4  5+- 6  7  8  9
    10 11+12 13 14 15+-16 17 18 19
    """

    graph = Graph()
    graph.add_nodes_from(list(range(42)))

    edges = []
    cols = 10
    rows = 2
    for i in range(rows):
        for j in range(cols):
            node = i * cols + j
            if j < cols - 1:
                edges.append((node, node + 1))
            if i < rows - 1:
                edges.append((node, node + cols))

    graph.add_edges_from(edges)

    for node in graph.nodes:
        graph.nodes[node]["TOTPOP"] = 1
        if node % 10 < 2:
            graph.nodes[node]["region"] = 0
            graph.nodes[node]["subregion"] = 0
        elif node % 10 < 6:
            graph.nodes[node]["region"] = 0
            graph.nodes[node]["subregion"] = 1
        else:
            graph.nodes[node]["region"] = 1
            graph.nodes[node]["subregion"] = 2

    for node in graph.nodes:
        graph.nodes[node]["district"] = 1 if node <= cols * rows // 2 else 2

    return graph


@pytest.fixture
def two_by_six_with_none_values():
    """
    0  1  2- 3  4  5
    6  7  8- 9 10 11
    """

    graph = Graph()
    graph.add_nodes_from(list(range(12)))

    edges = []
    cols = 6
    rows = 2
    for i in range(rows):
        for j in range(cols):
            node = i * cols + j
            if j < cols - 1:
                edges.append((node, node + 1))
            if i < rows - 1:
                edges.append((node, node + cols))

    graph.add_edges_from(edges)

    for node in graph.nodes:
        graph.nodes[node]["TOTPOP"] = 1
        if node % 6 < 3:
            graph.nodes[node]["region"] = 0
        else:
            graph.nodes[node]["region"] = None

    for node in graph.nodes:
        graph.nodes[node]["district"] = 1 if node <= cols * rows // 2 else 2

    return graph


def test_partition_heating_graph_correctly_single_edge(seven_by_eight_grid_single_join):

    graph = add_surcharges(
        seven_by_eight_grid_single_join,
        region_surcharge={"region": 0.9},
        attenuation_factor=0.5,
        attenuation_radius=2,
    )

    my_partition = Partition(
        graph,
        assignment="district",
    )

    # fmt: off
    weighted_edges = set(
        [
            (11, 19, frozendict({"surcharge": 0.9 * 0.5**2, "priority": 1})),
            (12, 20, frozendict({"surcharge": 0.9 * 0.5**2, "priority": 1})),
            (18, 19, frozendict({"surcharge": 0.9 * 0.5**2, "priority": 1})),
            (18, 26, frozendict({"surcharge": 0.9 * 0.5**2, "priority": 1})),
            (19, 27, frozendict({"surcharge": 0.9 * 0.5**1, "priority": 1})),
            (20, 28, frozendict({"surcharge": 0.9 * 0.5**1, "priority": 1})),
            (20, 21, frozendict({"surcharge": 0.9 * 0.5**2, "priority": 1})),
            (21, 29, frozendict({"surcharge": 0.9 * 0.5**2, "priority": 1})),
            (25, 26, frozendict({"surcharge": 0.9 * 0.5**2, "priority": 1})),
            (26, 27, frozendict({"surcharge": 0.9 * 0.5**1, "priority": 1})),
            (26, 34, frozendict({"surcharge": 0.9 * 0.5**2, "priority": 1})),
            (27, 28, frozendict({"surcharge": 0.9 * 0.5**0, "priority": 0})),
            (27, 35, frozendict({"surcharge": 0.9 * 0.5**1, "priority": 1})),
            (28, 36, frozendict({"surcharge": 0.9 * 0.5**1, "priority": 1})),
            (28, 29, frozendict({"surcharge": 0.9 * 0.5**1, "priority": 1})),
            (29, 37, frozendict({"surcharge": 0.9 * 0.5**2, "priority": 1})),
            (29, 30, frozendict({"surcharge": 0.9 * 0.5**2, "priority": 1})),
            (34, 35, frozendict({"surcharge": 0.9 * 0.5**2, "priority": 1})),
            (35, 43, frozendict({"surcharge": 0.9 * 0.5**2, "priority": 1})),
            (36, 44, frozendict({"surcharge": 0.9 * 0.5**2, "priority": 1})),
            (36, 37, frozendict({"surcharge": 0.9 * 0.5**2, "priority": 1})),
        ]
    )
    # fmt: on

    for edge in weighted_edges:
        assert edge in my_partition.graph.edges(data=True)

    for edge in my_partition.graph.edges(data=True):
        tuple_edge = (edge[0], edge[1], frozendict(edge[2]))
        if tuple_edge not in weighted_edges:
            assert edge[2]["surcharge"] == 0.0
            assert edge[2]["priority"] == 1


def test_partition_heating_bipartition_correctly(four_by_eight_grid):

    graph = add_surcharges(
        four_by_eight_grid,
        region_surcharge={"region": 0.9},
        attenuation_factor=0.5,
        attenuation_radius=2,
    )

    my_partition = Partition(
        graph,
        assignment="district",
    )

    # fmt: off
    weighted_edges = set(
        [
            (1,  2,  frozendict({"surcharge": 0.9 * 0.5**2, "priority": 1})),
            (2,  3,  frozendict({"surcharge": 0.9 * 0.5**1, "priority": 1})),
            (2,  10, frozendict({"surcharge": 0.9 * 0.5**2, "priority": 1})),
            (3,  4,  frozendict({"surcharge": 0.9 * 0.5**0, "priority": 0})),
            (3,  11, frozendict({"surcharge": 0.9 * 0.5**1, "priority": 1})),
            (4,  5,  frozendict({"surcharge": 0.9 * 0.5**1, "priority": 1})),
            (4,  12, frozendict({"surcharge": 0.9 * 0.5**1, "priority": 1})),
            (5,  6,  frozendict({"surcharge": 0.9 * 0.5**2, "priority": 1})),
            (5,  13, frozendict({"surcharge": 0.9 * 0.5**2, "priority": 1})),
            (9,  10, frozendict({"surcharge": 0.9 * 0.5**2, "priority": 1})),
            (10, 11, frozendict({"surcharge": 0.9 * 0.5**1, "priority": 1})),
            (10, 18, frozendict({"surcharge": 0.9 * 0.5**2, "priority": 1})),
            (11, 12, frozendict({"surcharge": 0.9 * 0.5**0, "priority": 0})),
            (11, 19, frozendict({"surcharge": 0.9 * 0.5**1, "priority": 1})),
            (12, 13, frozendict({"surcharge": 0.9 * 0.5**1, "priority": 1})),
            (12, 20, frozendict({"surcharge": 0.9 * 0.5**1, "priority": 1})),
            (13, 14, frozendict({"surcharge": 0.9 * 0.5**2, "priority": 1})),
            (13, 21, frozendict({"surcharge": 0.9 * 0.5**2, "priority": 1})),
            (17, 18, frozendict({"surcharge": 0.9 * 0.5**2, "priority": 1})),
            (18, 19, frozendict({"surcharge": 0.9 * 0.5**1, "priority": 1})),
            (18, 26, frozendict({"surcharge": 0.9 * 0.5**2, "priority": 1})),
            (19, 20, frozendict({"surcharge": 0.9 * 0.5**0, "priority": 0})),
            (19, 27, frozendict({"surcharge": 0.9 * 0.5**1, "priority": 1})),
            (20, 21, frozendict({"surcharge": 0.9 * 0.5**1, "priority": 1})),
            (20, 28, frozendict({"surcharge": 0.9 * 0.5**1, "priority": 1})),
            (21, 22, frozendict({"surcharge": 0.9 * 0.5**2, "priority": 1})),
            (21, 29, frozendict({"surcharge": 0.9 * 0.5**2, "priority": 1})),
            (25, 26, frozendict({"surcharge": 0.9 * 0.5**2, "priority": 1})),
            (26, 27, frozendict({"surcharge": 0.9 * 0.5**1, "priority": 1})),
            (27, 28, frozendict({"surcharge": 0.9 * 0.5**0, "priority": 0})),
            (28, 29, frozendict({"surcharge": 0.9 * 0.5**1, "priority": 1})),
            (29, 30, frozendict({"surcharge": 0.9 * 0.5**2, "priority": 1})),
        ]
    )
    # fmt: on

    for edge in weighted_edges:
        assert edge in my_partition.graph.edges(data=True)

    for edge in my_partition.graph.edges(data=True):
        tuple_edge = (edge[0], edge[1], frozendict(edge[2]))
        if tuple_edge not in weighted_edges:
            assert edge[2]["surcharge"] == 0.0
            assert edge[2]["priority"] == 1


def test_partition_heating_triparition_small_radius_correctly(
    three_by_nine_grid_tripartition,
):
    graph = add_surcharges(
        three_by_nine_grid_tripartition,
        region_surcharge={"region": 0.9},
        attenuation_factor=0.5,
        attenuation_radius=1,
    )

    my_partition = Partition(
        graph,
        assignment="district",
    )

    # fmt: off
    weighted_edges = set(
        [
            (1,  2,  frozendict({"surcharge": 0.9 * 0.5**1, "priority": 1})),
            (2,  3,  frozendict({"surcharge": 0.9 * 0.5**0, "priority": 0})),
            (2,  11, frozendict({"surcharge": 0.9 * 0.5**1, "priority": 1})),
            (3,  4,  frozendict({"surcharge": 0.9 * 0.5**1, "priority": 1})),
            (3,  12, frozendict({"surcharge": 0.9 * 0.5**1, "priority": 1})),
            (4,  5,  frozendict({"surcharge": 0.9 * 0.5**1, "priority": 1})),
            (5,  6,  frozendict({"surcharge": 0.9 * 0.5**0, "priority": 0})),
            (5,  14, frozendict({"surcharge": 0.9 * 0.5**1, "priority": 1})),
            (6,  7,  frozendict({"surcharge": 0.9 * 0.5**1, "priority": 1})),
            (6,  15, frozendict({"surcharge": 0.9 * 0.5**1, "priority": 1})),
            (10, 11, frozendict({"surcharge": 0.9 * 0.5**1, "priority": 1})),
            (11, 12, frozendict({"surcharge": 0.9 * 0.5**0, "priority": 0})),
            (11, 20, frozendict({"surcharge": 0.9 * 0.5**1, "priority": 1})),
            (12, 13, frozendict({"surcharge": 0.9 * 0.5**1, "priority": 1})),
            (12, 21, frozendict({"surcharge": 0.9 * 0.5**1, "priority": 1})),
            (13, 14, frozendict({"surcharge": 0.9 * 0.5**1, "priority": 1})),
            (14, 15, frozendict({"surcharge": 0.9 * 0.5**0, "priority": 0})),
            (14, 23, frozendict({"surcharge": 0.9 * 0.5**1, "priority": 1})),
            (15, 16, frozendict({"surcharge": 0.9 * 0.5**1, "priority": 1})),
            (15, 24, frozendict({"surcharge": 0.9 * 0.5**1, "priority": 1})),
            (19, 20, frozendict({"surcharge": 0.9 * 0.5**1, "priority": 1})),
            (20, 21, frozendict({"surcharge": 0.9 * 0.5**0, "priority": 0})),
            (21, 22, frozendict({"surcharge": 0.9 * 0.5**1, "priority": 1})),
            (22, 23, frozendict({"surcharge": 0.9 * 0.5**1, "priority": 1})),
            (23, 24, frozendict({"surcharge": 0.9 * 0.5**0, "priority": 0})),
            (24, 25, frozendict({"surcharge": 0.9 * 0.5**1, "priority": 1})),
        ]
    )
    # fmt: on

    for edge in weighted_edges:
        assert edge in my_partition.graph.edges(data=True)

    for edge in my_partition.graph.edges(data=True):
        tuple_edge = (edge[0], edge[1], frozendict(edge[2]))
        if tuple_edge not in weighted_edges:
            assert edge[2]["surcharge"] == 0.0
            assert edge[2]["priority"] == 1


def test_partition_heating_triparition_large_radius_correctly(
    three_by_nine_grid_tripartition,
):
    graph = add_surcharges(
        three_by_nine_grid_tripartition,
        region_surcharge={"region": 0.9},
        attenuation_factor=0.5,
        attenuation_radius=10,
    )

    my_partition = Partition(
        graph,
        assignment="district",
    )

    # fmt: off
    weighted_edges = set(
        [
            (0,  1,  frozendict({"surcharge": 0.9 * 0.5**2, "priority": 1})),
            (0,  9,  frozendict({"surcharge": 0.9 * 0.5**3, "priority": 1})),
            (1,  2,  frozendict({"surcharge": 0.9 * 0.5**1, "priority": 1})),
            (1,  10, frozendict({"surcharge": 0.9 * 0.5**2, "priority": 1})),
            (2,  3,  frozendict({"surcharge": 0.9 * 0.5**0, "priority": 0})),
            (2,  11, frozendict({"surcharge": 0.9 * 0.5**1, "priority": 1})),
            (3,  4,  frozendict({"surcharge": 0.9 * 0.5**1, "priority": 1})),
            (3,  12, frozendict({"surcharge": 0.9 * 0.5**1, "priority": 1})),
            (4,  5,  frozendict({"surcharge": 0.9 * 0.5**1, "priority": 1})),
            (4,  13, frozendict({"surcharge": 0.9 * 0.5**2, "priority": 1})),
            (5,  6,  frozendict({"surcharge": 0.9 * 0.5**0, "priority": 0})),
            (5,  14, frozendict({"surcharge": 0.9 * 0.5**1, "priority": 1})),
            (6,  7,  frozendict({"surcharge": 0.9 * 0.5**1, "priority": 1})),
            (6,  15, frozendict({"surcharge": 0.9 * 0.5**1, "priority": 1})),
            (7,  8,  frozendict({"surcharge": 0.9 * 0.5**2, "priority": 1})),
            (7,  16, frozendict({"surcharge": 0.9 * 0.5**2, "priority": 1})),
            (8,  17, frozendict({"surcharge": 0.9 * 0.5**3, "priority": 1})),
            (9,  10, frozendict({"surcharge": 0.9 * 0.5**2, "priority": 1})),
            (9,  18, frozendict({"surcharge": 0.9 * 0.5**3, "priority": 1})),
            (10, 11, frozendict({"surcharge": 0.9 * 0.5**1, "priority": 1})),
            (10, 19, frozendict({"surcharge": 0.9 * 0.5**2, "priority": 1})),
            (11, 12, frozendict({"surcharge": 0.9 * 0.5**0, "priority": 0})),
            (11, 20, frozendict({"surcharge": 0.9 * 0.5**1, "priority": 1})),
            (12, 13, frozendict({"surcharge": 0.9 * 0.5**1, "priority": 1})),
            (12, 21, frozendict({"surcharge": 0.9 * 0.5**1, "priority": 1})),
            (13, 14, frozendict({"surcharge": 0.9 * 0.5**1, "priority": 1})),
            (13, 22, frozendict({"surcharge": 0.9 * 0.5**2, "priority": 1})),
            (14, 15, frozendict({"surcharge": 0.9 * 0.5**0, "priority": 0})),
            (14, 23, frozendict({"surcharge": 0.9 * 0.5**1, "priority": 1})),
            (15, 16, frozendict({"surcharge": 0.9 * 0.5**1, "priority": 1})),
            (15, 24, frozendict({"surcharge": 0.9 * 0.5**1, "priority": 1})),
            (16, 17, frozendict({"surcharge": 0.9 * 0.5**2, "priority": 1})),
            (16, 25, frozendict({"surcharge": 0.9 * 0.5**2, "priority": 1})),
            (17, 26, frozendict({"surcharge": 0.9 * 0.5**3, "priority": 1})),
            (18, 19, frozendict({"surcharge": 0.9 * 0.5**2, "priority": 1})),
            (19, 20, frozendict({"surcharge": 0.9 * 0.5**1, "priority": 1})),
            (20, 21, frozendict({"surcharge": 0.9 * 0.5**0, "priority": 0})),
            (21, 22, frozendict({"surcharge": 0.9 * 0.5**1, "priority": 1})),
            (22, 23, frozendict({"surcharge": 0.9 * 0.5**1, "priority": 1})),
            (23, 24, frozendict({"surcharge": 0.9 * 0.5**0, "priority": 0})),
            (24, 25, frozendict({"surcharge": 0.9 * 0.5**1, "priority": 1})),
            (25, 26, frozendict({"surcharge": 0.9 * 0.5**2, "priority": 1})),
        ]
    )
    # fmt: on

    for edge in weighted_edges:
        assert edge in my_partition.graph.edges(data=True)


def test_partition_heating_different_radii_correctly_overlap(
    two_by_ten_grid_subregion_overlap,
):
    reg = 1.0
    sub = 0.9

    graph = add_surcharges(
        two_by_ten_grid_subregion_overlap,
        region_surcharge={"region": reg, "subregion": sub},
        attenuation_factor=0.5,
        attenuation_radius={"region": 2, "subregion": 1},
    )

    my_partition = Partition(
        graph,
        assignment="district",
    )

    # fmt: off
    weighted_edges = set(
        [
            (0, 1,   frozendict({"surcharge": max(0.0         , sub * 0.5**1), "priority": 2})),
            (1, 2,   frozendict({"surcharge": max(0.0         , sub * 0.5**0), "priority": 1})),
            (1, 11,  frozendict({"surcharge": max(0.0         , sub * 0.5**1), "priority": 2})),
            (2, 3,   frozendict({"surcharge": max(0.0         , sub * 0.5**1), "priority": 2})),
            (2, 12,  frozendict({"surcharge": max(0.0         , sub * 0.5**1), "priority": 2})),
            (3, 4,   frozendict({"surcharge": max(reg * 0.5**2, 0.0         ), "priority": 2})),
            (4, 5,   frozendict({"surcharge": max(reg * 0.5**1, sub * 0.5**1), "priority": 2})),
            (4, 14,  frozendict({"surcharge": max(reg * 0.5**2, 0.0         ), "priority": 2})),
            (5, 6,   frozendict({"surcharge": max(reg * 0.5**0, sub * 0.5**0), "priority": 0})),
            (5, 15,  frozendict({"surcharge": max(reg * 0.5**1, sub * 0.5**1), "priority": 2})),
            (6, 7,   frozendict({"surcharge": max(reg * 0.5**1, sub * 0.5**1), "priority": 2})),
            (6, 16,  frozendict({"surcharge": max(reg * 0.5**1, sub * 0.5**1), "priority": 2})),
            (7, 8,   frozendict({"surcharge": max(reg * 0.5**2, 0.0         ), "priority": 2})),
            (7, 17,  frozendict({"surcharge": max(reg * 0.5**2, 0.0         ), "priority": 2})),
            (10, 11, frozendict({"surcharge": max(0.0         , sub * 0.5**1), "priority": 2})),
            (11, 12, frozendict({"surcharge": max(0.0         , sub * 0.5**0), "priority": 1})),
            (12, 13, frozendict({"surcharge": max(0.0         , sub * 0.5**1), "priority": 2})),
            (13, 14, frozendict({"surcharge": max(reg * 0.5**2, 0.0         ), "priority": 2})),
            (14, 15, frozendict({"surcharge": max(reg * 0.5**1, sub * 0.5**1), "priority": 2})),
            (15, 16, frozendict({"surcharge": max(reg * 0.5**0, sub * 0.5**0), "priority": 0})),
            (16, 17, frozendict({"surcharge": max(reg * 0.5**1, sub * 0.5**1), "priority": 2})),
            (17, 18, frozendict({"surcharge": max(reg * 0.5**2, 0.0         ), "priority": 2})),
        ]
    )
    # fmt: on

    for edge in weighted_edges:
        assert edge in my_partition.graph.edges(data=True)

    for edge in my_partition.graph.edges(data=True):
        tuple_edge = (edge[0], edge[1], frozendict(edge[2]))
        if tuple_edge not in weighted_edges:
            assert edge[2]["surcharge"] == 0.0
            assert edge[2]["priority"] == 2


def test_partition_heating_different_radii_correctly_overlap_many_weights(
    two_by_ten_grid_subregion_overlap,
):
    for reg in [x / 10 for x in range(1, 11)]:
        for sub in [x / 10 for x in range(1, 11)]:
            graph = add_surcharges(
                two_by_ten_grid_subregion_overlap.copy(),
                region_surcharge={"region": reg, "subregion": sub},
                attenuation_factor=0.5,
                attenuation_radius={"region": 2, "subregion": 1},
            )

            my_partition = Partition(
                graph,
                assignment="district",
            )

            # fmt: off
            weighted_edges = set(
                [
                    (0,  1,  frozendict({"surcharge": max(0.0         , sub * 0.5**1), "priority": 2}),),
                    (1,  2,  frozendict({"surcharge": max(0.0         , sub * 0.5**0), "priority": 1}),),
                    (1,  11, frozendict({"surcharge": max(0.0         , sub * 0.5**1), "priority": 2}),),
                    (2,  3,  frozendict({"surcharge": max(0.0         , sub * 0.5**1), "priority": 2}),),
                    (2,  12, frozendict({"surcharge": max(0.0         , sub * 0.5**1), "priority": 2}),),
                    (3,  4,  frozendict({"surcharge": max(reg * 0.5**2, 0.0         ), "priority": 2}),),
                    (4,  5,  frozendict({"surcharge": max(reg * 0.5**1, sub * 0.5**1), "priority": 2}),),
                    (4,  14, frozendict({"surcharge": max(reg * 0.5**2, 0.0         ), "priority": 2}),),
                    (5,  6,  frozendict({"surcharge": max(reg * 0.5**0, sub * 0.5**0), "priority": 0}),),
                    (5,  15, frozendict({"surcharge": max(reg * 0.5**1, sub * 0.5**1), "priority": 2}),),
                    (6,  7,  frozendict({"surcharge": max(reg * 0.5**1, sub * 0.5**1), "priority": 2}),),
                    (6,  16, frozendict({"surcharge": max(reg * 0.5**1, sub * 0.5**1), "priority": 2}),),
                    (7,  8,  frozendict({"surcharge": max(reg * 0.5**2, 0.0         ), "priority": 2}),),
                    (7,  17, frozendict({"surcharge": max(reg * 0.5**2, 0.0         ), "priority": 2}),),
                    (10, 11, frozendict({"surcharge": max(0.0         , sub * 0.5**1), "priority": 2}),),
                    (11, 12, frozendict({"surcharge": max(0.0         , sub * 0.5**0), "priority": 1}),),
                    (12, 13, frozendict({"surcharge": max(0.0         , sub * 0.5**1), "priority": 2}),),
                    (13, 14, frozendict({"surcharge": max(reg * 0.5**2, 0.0         ), "priority": 2}),),
                    (14, 15, frozendict({"surcharge": max(reg * 0.5**1, sub * 0.5**1), "priority": 2}),),
                    (15, 16, frozendict({"surcharge": max(reg * 0.5**0, sub * 0.5**0), "priority": 0}),),
                    (16, 17, frozendict({"surcharge": max(reg * 0.5**1, sub * 0.5**1), "priority": 2}),),
                    (17, 18, frozendict({"surcharge": max(reg * 0.5**2, 0.0         ), "priority": 2}),),
                ]
            )
            # fmt: on

            for edge in weighted_edges:
                assert edge in my_partition.graph.edges(data=True)

            for edge in my_partition.graph.edges(data=True):
                tuple_edge = (edge[0], edge[1], frozendict(edge[2]))
                if tuple_edge not in weighted_edges:
                    assert edge[2]["surcharge"] == 0.0
                    assert edge[2]["priority"] == 2


def test_partition_heating_different_radii_correctly_distinct(
    two_by_ten_grid_subregion_distinct,
):
    reg = 1.0
    sub = 0.9

    graph = add_surcharges(
        two_by_ten_grid_subregion_distinct,
        region_surcharge={"region": reg, "subregion": sub},
        attenuation_factor=0.5,
        attenuation_radius={"region": 2, "subregion": 1},
    )

    my_partition = Partition(
        graph,
        assignment="district",
    )

    # fmt: off
    weighted_edges = set(
        [
            (0, 1,   frozendict({"surcharge": max(0.0         , sub * 0.5**1), "priority": 2})),
            (1, 2,   frozendict({"surcharge": max(0.0         , sub * 0.5**0), "priority": 1})),
            (1, 11,  frozendict({"surcharge": max(0.0         , sub * 0.5**1), "priority": 2})),
            (2, 3,   frozendict({"surcharge": max(reg * 0.5**2, sub * 0.5**1), "priority": 2})),
            (2, 12,  frozendict({"surcharge": max(0.0         , sub * 0.5**1), "priority": 2})),
            (3, 4,   frozendict({"surcharge": max(reg * 0.5**1, 0.0         ), "priority": 2})),
            (3, 13,  frozendict({"surcharge": max(reg * 0.5**2, 0.0         ), "priority": 2})),
            (4, 5,   frozendict({"surcharge": max(reg * 0.5**0, sub * 0.5**1), "priority": 1})),
            (4, 14,  frozendict({"surcharge": max(reg * 0.5**1, 0.0         ), "priority": 2})),
            (5, 6,   frozendict({"surcharge": max(reg * 0.5**1, sub * 0.5**0), "priority": 1})),
            (5, 15,  frozendict({"surcharge": max(reg * 0.5**1, sub * 0.5**1), "priority": 2})),
            (6, 7,   frozendict({"surcharge": max(reg * 0.5**2, sub * 0.5**1), "priority": 2})),
            (6, 16,  frozendict({"surcharge": max(reg * 0.5**2, sub * 0.5**1), "priority": 2})),
            (10, 11, frozendict({"surcharge": max(0.0         , sub * 0.5**1), "priority": 2})),
            (11, 12, frozendict({"surcharge": max(0.0         , sub * 0.5**0), "priority": 1})),
            (12, 13, frozendict({"surcharge": max(reg * 0.5**2, sub * 0.5**1), "priority": 2})),
            (13, 14, frozendict({"surcharge": max(reg * 0.5**1, 0.0         ), "priority": 2})),
            (14, 15, frozendict({"surcharge": max(reg * 0.5**0, sub * 0.5**1), "priority": 1})),
            (15, 16, frozendict({"surcharge": max(reg * 0.5**1, sub * 0.5**0), "priority": 1})),
            (16, 17, frozendict({"surcharge": max(reg * 0.5**2, sub * 0.5**1), "priority": 2})),
        ]
    )
    # fmt: on

    for edge in weighted_edges:
        assert edge in my_partition.graph.edges(data=True)

    for edge in my_partition.graph.edges(data=True):
        tuple_edge = (edge[0], edge[1], frozendict(edge[2]))
        if tuple_edge not in weighted_edges:
            assert edge[2]["surcharge"] == 0.0
            assert edge[2]["priority"] == 2


def test_partition_heating_different_radii_correctly_distinct_many_weights(
    two_by_ten_grid_subregion_distinct,
):

    reg = 1.0
    sub = 0.9
    for reg in [x / 10 for x in range(1, 11)]:
        for sub in [x / 10 for x in range(1, 11)]:
            graph = add_surcharges(
                two_by_ten_grid_subregion_distinct.copy(),
                region_surcharge={"region": reg, "subregion": sub},
                attenuation_factor=0.5,
                attenuation_radius={"region": 2, "subregion": 1},
            )

            my_partition = Partition(
                graph,
                assignment="district",
            )

            # fmt: off
            weighted_edges = set(
                [
                    (0, 1,   frozendict({"surcharge": max(0.0         , sub * 0.5**1), "priority": 2})),
                    (1, 2,   frozendict({"surcharge": max(0.0         , sub * 0.5**0), "priority": 1})),
                    (1, 11,  frozendict({"surcharge": max(0.0         , sub * 0.5**1), "priority": 2})),
                    (2, 3,   frozendict({"surcharge": max(reg * 0.5**2, sub * 0.5**1), "priority": 2})),
                    (2, 12,  frozendict({"surcharge": max(0.0         , sub * 0.5**1), "priority": 2})),
                    (3, 4,   frozendict({"surcharge": max(reg * 0.5**1, 0.0         ), "priority": 2})),
                    (3, 13,  frozendict({"surcharge": max(reg * 0.5**2, 0.0         ), "priority": 2})),
                    (4, 5,   frozendict({"surcharge": max(reg * 0.5**0, sub * 0.5**1), "priority": 1})),
                    (4, 14,  frozendict({"surcharge": max(reg * 0.5**1, 0.0         ), "priority": 2})),
                    (5, 6,   frozendict({"surcharge": max(reg * 0.5**1, sub * 0.5**0), "priority": 1})),
                    (5, 15,  frozendict({"surcharge": max(reg * 0.5**1, sub * 0.5**1), "priority": 2})),
                    (6, 7,   frozendict({"surcharge": max(reg * 0.5**2, sub * 0.5**1), "priority": 2})),
                    (6, 16,  frozendict({"surcharge": max(reg * 0.5**2, sub * 0.5**1), "priority": 2})),
                    (10, 11, frozendict({"surcharge": max(0.0         , sub * 0.5**1), "priority": 2})),
                    (11, 12, frozendict({"surcharge": max(0.0         , sub * 0.5**0), "priority": 1})),
                    (12, 13, frozendict({"surcharge": max(reg * 0.5**2, sub * 0.5**1), "priority": 2})),
                    (13, 14, frozendict({"surcharge": max(reg * 0.5**1, 0.0         ), "priority": 2})),
                    (14, 15, frozendict({"surcharge": max(reg * 0.5**0, sub * 0.5**1), "priority": 1})),
                    (15, 16, frozendict({"surcharge": max(reg * 0.5**1, sub * 0.5**0), "priority": 1})),
                    (16, 17, frozendict({"surcharge": max(reg * 0.5**2, sub * 0.5**1), "priority": 2})),
                ]
            )
            # fmt: on

            for edge in weighted_edges:
                assert edge in my_partition.graph.edges(data=True)

            for edge in my_partition.graph.edges(data=True):
                tuple_edge = (edge[0], edge[1], frozendict(edge[2]))
                if tuple_edge not in weighted_edges:
                    assert edge[2]["surcharge"] == 0.0
                    assert edge[2]["priority"] == 2


def test_partition_heating_none_values_correctly(two_by_six_with_none_values):
    reg = 1.0
    sub = 0.9

    graph = add_surcharges(
        two_by_six_with_none_values,
        region_surcharge={"region": reg},
        attenuation_factor=0.5,
        attenuation_radius={"region": 1},
    )

    my_partition = Partition(
        graph,
        assignment="district",
    )

    # fmt: off
    weighted_edges = set(
        [
            (0,  6,  frozendict({'surcharge': 0.0, 'priority': 1})),
            (1,  2,  frozendict({'surcharge': 0.5, 'priority': 1})),
            (1,  7,  frozendict({'surcharge': 0.0, 'priority': 1})),
            (2,  3,  frozendict({'surcharge': 1.0, 'priority': 0})),
            (2,  8,  frozendict({'surcharge': 0.5, 'priority': 1})),
            (3,  4,  frozendict({'surcharge': 1.0, 'priority': 0})),
            (3,  9,  frozendict({'surcharge': 1.0, 'priority': 0})),
            (4,  5,  frozendict({'surcharge': 1.0, 'priority': 0})),
            (4,  10, frozendict({'surcharge': 1.0, 'priority': 0})),
            (5,  11, frozendict({'surcharge': 1.0, 'priority': 0})),
            (6,  7,  frozendict({'surcharge': 0.0, 'priority': 1})),
            (7,  8,  frozendict({'surcharge': 0.5, 'priority': 1})),
            (8,  9,  frozendict({'surcharge': 1.0, 'priority': 0})),
            (9,  10, frozendict({'surcharge': 1.0, 'priority': 0})),
            (10, 11, frozendict({'surcharge': 1.0, 'priority': 0})),
        ]
    )
    # fmt: on

    for edge in weighted_edges:
        assert edge in my_partition.graph.edges(data=True)


def test_partition_heating_one_coi_with_gap_between_different_labels(
    three_by_nine_grid_tripartition,
):
    graph = add_surcharges(
        three_by_nine_grid_tripartition,
        region_surcharge={"coi_3": 0.9},
        attenuation_factor=0.5,
        attenuation_radius=1,
    )

    my_partition = Partition(
        graph,
        assignment="district",
    )

    # fmt: off
    weighted_edges = set(
        [
            (0,  1,  frozendict({'surcharge': 0.0,  'priority': 1.0})),
            (0,  9,  frozendict({'surcharge': 0.0,  'priority': 1.0})),
            (1,  2,  frozendict({'surcharge': 0.45, 'priority': 1.0})),
            (1,  10, frozendict({'surcharge': 0.0,  'priority': 1.0})),
            (2,  3,  frozendict({'surcharge': 0.9,  'priority': 0.0})),
            (2,  11, frozendict({'surcharge': 0.45, 'priority': 1.0})),
            (3,  4,  frozendict({'surcharge': 0.9,  'priority': 0.0})),
            (3,  12, frozendict({'surcharge': 0.9,  'priority': 0.0})),
            (4,  5,  frozendict({'surcharge': 0.9,  'priority': 0.0})),
            (4,  13, frozendict({'surcharge': 0.9,  'priority': 0.0})),
            (5,  6,  frozendict({'surcharge': 0.9,  'priority': 0.0})),
            (5,  14, frozendict({'surcharge': 0.9,  'priority': 0.0})),
            (6,  7,  frozendict({'surcharge': 0.45, 'priority': 1.0})),
            (6,  15, frozendict({'surcharge': 0.45, 'priority': 1.0})),
            (7,  8,  frozendict({'surcharge': 0.0,  'priority': 1.0})),
            (7,  16, frozendict({'surcharge': 0.0,  'priority': 1.0})),
            (8,  17, frozendict({'surcharge': 0.0,  'priority': 1.0})),
            (9,  10, frozendict({'surcharge': 0.0,  'priority': 1.0})),
            (9,  18, frozendict({'surcharge': 0.0,  'priority': 1.0})),
            (10, 11, frozendict({'surcharge': 0.45, 'priority': 1.0})),
            (10, 19, frozendict({'surcharge': 0.0,  'priority': 1.0})),
            (11, 12, frozendict({'surcharge': 0.9,  'priority': 0.0})),
            (11, 20, frozendict({'surcharge': 0.45, 'priority': 1.0})),
            (12, 13, frozendict({'surcharge': 0.9,  'priority': 0.0})),
            (12, 21, frozendict({'surcharge': 0.9,  'priority': 0.0})),
            (13, 14, frozendict({'surcharge': 0.9,  'priority': 0.0})),
            (13, 22, frozendict({'surcharge': 0.9,  'priority': 0.0})),
            (14, 15, frozendict({'surcharge': 0.9,  'priority': 0.0})),
            (14, 23, frozendict({'surcharge': 0.9,  'priority': 0.0})),
            (15, 16, frozendict({'surcharge': 0.45, 'priority': 1.0})),
            (15, 24, frozendict({'surcharge': 0.45, 'priority': 1.0})),
            (16, 17, frozendict({'surcharge': 0.0,  'priority': 1.0})),
            (16, 25, frozendict({'surcharge': 0.0,  'priority': 1.0})),
            (17, 26, frozendict({'surcharge': 0.0,  'priority': 1.0})),
            (18, 19, frozendict({'surcharge': 0.0,  'priority': 1.0})),
            (19, 20, frozendict({'surcharge': 0.45, 'priority': 1.0})),
            (20, 21, frozendict({'surcharge': 0.9,  'priority': 0.0})),
            (21, 22, frozendict({'surcharge': 0.9,  'priority': 0.0})),
            (22, 23, frozendict({'surcharge': 0.9,  'priority': 0.0})),
            (23, 24, frozendict({'surcharge': 0.9,  'priority': 0.0})),
            (24, 25, frozendict({'surcharge': 0.45, 'priority': 1.0})),
            (25, 26, frozendict({'surcharge': 0.0,  'priority': 1.0})),
        ]
    )
    # fmt: on

    for edge in weighted_edges:
        assert edge in my_partition.graph.edges(data=True)


def test_partition_heating_multiple_coi_with_none_values_correctly(
    three_by_nine_grid_tripartition,
):
    coi_1 = 1.0
    coi_2 = 0.9

    graph = add_surcharges(
        three_by_nine_grid_tripartition,
        region_surcharge={"coi_1": coi_1, "coi_2": coi_2},
        attenuation_factor=0.5,
        attenuation_radius={"coi_1": 1, "coi_2": 2},
    )

    my_partition = Partition(
        graph,
        assignment="district",
    )

    # fmt: off
    weighted_edges = set(
        [   
            (0,  1,  frozendict({'surcharge': 0.9, 'priority': 1.0})),
            (0,  9,  frozendict({'surcharge': 0.9, 'priority': 1.0})),
            (1,  2,  frozendict({'surcharge': 0.9, 'priority': 1.0})),
            (1,  10, frozendict({'surcharge': 0.9, 'priority': 1.0})),
            (2,  3,  frozendict({'surcharge': 1.0, 'priority': 0.0})),
            (2,  11, frozendict({'surcharge': 0.9, 'priority': 1.0})),
            (3,  4,  frozendict({'surcharge': 1.9, 'priority': 0.0})),
            (3,  12, frozendict({'surcharge': 1.9, 'priority': 0.0})),
            (4,  5,  frozendict({'surcharge': 1.9, 'priority': 0.0})),
            (4,  13, frozendict({'surcharge': 1.9, 'priority': 0.0})),
            (5,  6,  frozendict({'surcharge': 1.0, 'priority': 0.0})),
            (5,  14, frozendict({'surcharge': 1.9, 'priority': 0.0})),
            (6,  7,  frozendict({'surcharge': 1.0, 'priority': 1.0})),
            (6,  15, frozendict({'surcharge': 1.0, 'priority': 1.0})),
            (7,  8,  frozendict({'surcharge': 1.0, 'priority': 1.0})),
            (7,  16, frozendict({'surcharge': 1.0, 'priority': 1.0})),
            (8,  17, frozendict({'surcharge': 1.0, 'priority': 1.0})),
            (9,  10, frozendict({'surcharge': 0.9, 'priority': 1.0})),
            (9,  18, frozendict({'surcharge': 0.9, 'priority': 1.0})),
            (10, 11, frozendict({'surcharge': 0.9, 'priority': 1.0})),
            (10, 19, frozendict({'surcharge': 0.9, 'priority': 1.0})),
            (11, 12, frozendict({'surcharge': 1.0, 'priority': 0.0})),
            (11, 20, frozendict({'surcharge': 0.9, 'priority': 1.0})),
            (12, 13, frozendict({'surcharge': 1.9, 'priority': 0.0})),
            (12, 21, frozendict({'surcharge': 1.9, 'priority': 0.0})),
            (13, 14, frozendict({'surcharge': 1.9, 'priority': 0.0})),
            (13, 22, frozendict({'surcharge': 1.9, 'priority': 0.0})),
            (14, 15, frozendict({'surcharge': 1.0, 'priority': 0.0})),
            (14, 23, frozendict({'surcharge': 1.9, 'priority': 0.0})),
            (15, 16, frozendict({'surcharge': 1.0, 'priority': 1.0})),
            (15, 24, frozendict({'surcharge': 1.0, 'priority': 1.0})),
            (16, 17, frozendict({'surcharge': 1.0, 'priority': 1.0})),
            (16, 25, frozendict({'surcharge': 1.0, 'priority': 1.0})),
            (17, 26, frozendict({'surcharge': 1.0, 'priority': 1.0})),
            (18, 19, frozendict({'surcharge': 0.9, 'priority': 1.0})),
            (19, 20, frozendict({'surcharge': 0.9, 'priority': 1.0})),
            (20, 21, frozendict({'surcharge': 1.0, 'priority': 0.0})),
            (21, 22, frozendict({'surcharge': 1.9, 'priority': 0.0})),
            (22, 23, frozendict({'surcharge': 1.9, 'priority': 0.0})),
            (23, 24, frozendict({'surcharge': 1.0, 'priority': 0.0})),
            (24, 25, frozendict({'surcharge': 1.0, 'priority': 1.0})),
            (25, 26, frozendict({'surcharge': 1.0, 'priority': 1.0})),
        ]
    )

    # fmt: on

    for edge in weighted_edges:
        assert edge in my_partition.graph.edges(data=True)
