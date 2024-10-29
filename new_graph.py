import functools
import json
from typing import Any
import warnings

import rustworkx
import networkx
from networkx.classes.function import frozen
from networkx.readwrite import json_graph
import pandas as pd

from gerrychain.graph.geo import GeometryError, invalid_geometries, reprojected
from gerrychain.graph.adjacency import neighbors

from typing import List, Iterable, Optional, Set, Tuple, Union

import inspect


def json_serialize(input_object: Any) -> Optional[int]:
    """
    This function is used to handle one of the common issues that
    appears when trying to convert a pandas dataframe into a JSON
    serializable object. Specifically, it handles the issue of converting
    the pandas int64 to a python int so that JSON can serialize it.
    This is specifically used so that we can write graphs out to JSON
    files.

    :param input_object: The object to be converted
    :type input_object: Any (expected to be a pd.Int64Dtype)

    :returns: The converted pandas object or None if input is not of type
        pd.Int64Dtype
    :rtype: Optional[int]
    """
    if pd.api.types.is_integer_dtype(input_object):  # handle int64
        return int(input_object)

    return None


class Graph(rustworkx.PyGraph):
    def __repr__(self):
        return "<Graph [{} nodes, {} edges]>".format(
            len(self.node_indices()), len(self.edge_indices())
        )

    def __new__(cls, multigraph=False):
        return super(Graph, cls).__new__(cls, multigraph, attrs=dict())

    def graph(self):
        """
        This is a dummy method to help improve the backwards compatibility
        of the Graph object. This returns the attributes of the main
        graph object.
        """
        warnings.warn(
            type=DeprecationWarning,
            message="Calling `Graph.graph` is will be deprecated in future versions "
            "of `GerryChain`. Use `Graph.attrs` instead.",
        )
        return self.attrs

    # Done
    @classmethod
    def from_networkx(cls, graph: networkx.Graph) -> "Graph":
        """
        Create a Graph instance from a networkx.Graph object.

        :param graph: The networkx graph to be converted.
        :type graph: networkx.Graph

        :returns: The converted graph as an instance of this class.
        :rtype: Graph
        """
        nx_graph = networkx.convert_node_labels_to_integers(graph)
        g = cls()
        g.add_nodes_from(nx_graph.nodes(data=True))
        g.add_edges_from([(u, v, data) for u, v, data in nx_graph.edges(data=True)])
        for key, value in nx_graph.graph.items():
            g.attrs[key] = value
        add_surcharges(g)
        return g

    # Done
    def to_networkx(self) -> networkx.Graph:
        new_nx = networkx.Graph()
        new_nx.add_nodes_from(self.nodes())
        new_nx.add_edges_from(
            list([(u, v, dict(data)) for u, v, data in self.weighted_edge_list()])
        )
        for key, value in self.attrs.items():
            if key != "data":
                new_nx.graph[key] = value
        return new_nx

    @classmethod
    def from_json(cls, json_file: str) -> "Graph":
        """
        Load a graph from a JSON file in the NetworkX json_graph format.

        :param json_file: Path to JSON file.
        :type json_file: str

        :returns: The loaded graph as an instance of this class.
        :rtype: Graph
        """
        with open(json_file) as f:
            data = json.load(f)
        g = json_graph.adjacency_graph(data)
        graph = cls.from_networkx(g)
        graph.issue_warnings()
        return graph

    def to_json(
        self,
        json_file: str,
        *,
        keep_surcharge_and_priority: bool = False,
        include_geometries_as_geojson: bool = False,
    ) -> None:
        data = json_graph.adjacency_data(self.to_networkx())

        for edge_data_list in data["adjacency"]:
            for edge_data in edge_data_list:
                if not keep_surcharge_and_priority:
                    edge_data.pop("surcharge", None)
                    edge_data.pop("priority", None)

        if include_geometries_as_geojson:
            convert_geometries_to_geojson(data)
        else:
            remove_geometries(data)

        with open(json_file, "w") as f:
            json.dump(data, f, default=json_serialize)

    @classmethod
    def from_file(
        cls,
        filename: str,
        adjacency: str = "rook",
        cols_to_add: Optional[List[str]] = None,
        reproject: bool = False,
        ignore_errors: bool = False,
    ) -> "Graph":
        """
        Create a :class:`Graph` from a shapefile (or GeoPackage, or GeoJSON, or
        any other library that :mod:`geopandas` can read. See :meth:`from_geodataframe`
        for more details.

        :param filename: Path to the shapefile / GeoPackage / GeoJSON / etc.
        :type filename: str
        :param adjacency: The adjacency type to use ("rook" or "queen"). Default is "rook"
        :type adjacency: str, optional
        :param cols_to_add: The names of the columns that you want to
            add to the graph as node attributes. Default is None.
        :type cols_to_add: Optional[List[str]], optional
        :param reproject: Whether to reproject to a UTM projection before
            creating the graph. Default is False.
        :type reproject: bool, optional
        :param ignore_errors: Whether to ignore all invalid geometries and try to continue
            creating the graph. Default is False.
        :type ignore_errors: bool, optional

        :returns: The Graph object of the geometries from `filename`.
        :rtype: Graph

        .. Warning::

            This method requires the optional ``geopandas`` dependency.
            So please install ``gerrychain`` with the ``geo`` extra
            via the command:

            .. code-block:: console

                pip install gerrychain[geo]

            or install ``geopandas`` separately.
        """
        import geopandas as gp

        df = gp.read_file(filename)
        graph = cls.from_geodataframe(
            df,
            adjacency=adjacency,
            cols_to_add=cols_to_add,
            reproject=reproject,
            ignore_errors=ignore_errors,
        )
        graph.attrs["crs"] = df.crs.to_json()
        return graph

    @classmethod
    def from_geodataframe(
        cls,
        dataframe: pd.DataFrame,
        adjacency: str = "rook",
        cols_to_add: Optional[List[str]] = None,
        reproject: bool = False,
        ignore_errors: bool = False,
        crs_override: Optional[Union[str, int]] = None,
        add_geometry_to_node_attrs: bool = False,
    ) -> "Graph":
        """
        Creates the adjacency :class:`Graph` of geometries described by `dataframe`.
        The areas of the polygons are included as node attributes (with key `area`).
        The shared perimeter of neighboring polygons are included as edge attributes
        (with key `shared_perim`).
        Nodes corresponding to polygons on the boundary of the union of all the geometries
        (e.g., the state, if your dataframe describes VTDs) have a `boundary_node` attribute
        (set to `True`) and a `boundary_perim` attribute with the length of this "exterior"
        boundary.

        By default, areas and lengths are computed in a UTM projection suitable for the
        geometries. This prevents the bizarro area and perimeter values that show up when
        you accidentally do computations in Longitude-Latitude coordinates. If the user
        specifies `reproject=False`, then the areas and lengths will be computed in the
        GeoDataFrame's current coordinate reference system. This option is for users who
        have a preferred CRS they would like to use.

        :param dataframe: The GeoDateFrame to convert
        :type dataframe: :class:`geopandas.GeoDataFrame`
        :param adjacency: The adjacency type to use ("rook" or "queen").
            Default is "rook".
        :type adjacency: str, optional
        :param cols_to_add: The names of the columns that you want to add to the graph as
            node attributes. Default is None which adds all columns.
        :type cols_to_add: Optional[List[str]], optional
        :param reproject: Whether to reproject to a UTM projection before
            creating the graph. Default is ``False``.
        :type reproject: bool, optional
        :param ignore_errors: Whether to ignore all invalid geometries and
            attept to create the graph anyway. Default is ``False``.
        :type ignore_errors: bool, optional
        :param crs_override: Value to override the CRS of the GeoDataFrame.
            Default is None.
        :type crs_override: Optional[Union[str,int]], optional

        :returns: The adjacency graph of the geometries from `dataframe`.
        :rtype: Graph
        """
        # Validate geometries before reprojection
        if not ignore_errors:
            invalid = invalid_geometries(dataframe)
            if len(invalid) > 0:
                raise GeometryError(
                    "Invalid geometries at rows {} before "
                    "reprojection. Consider repairing the affected geometries with "
                    "`.buffer(0)`, or pass `ignore_errors=True` to attempt to create "
                    "the graph anyways.".format(invalid)
                )

        # Project the dataframe to an appropriate UTM projection unless
        # explicitly told not to.
        if reproject:
            df = reprojected(dataframe)
            if ignore_errors:
                invalid_reproj = invalid_geometries(df)
                print(invalid_reproj)
                if len(invalid_reproj) > 0:
                    raise GeometryError(
                        "Invalid geometries at rows {} after "
                        "reprojection. Consider reloading the GeoDataFrame with "
                        "`reproject=False` or repairing the affected geometries "
                        "with `.buffer(0)`.".format(invalid_reproj)
                    )
        else:
            df = dataframe

        # Generate dict of dicts of dicts with shared perimeters according
        # to the requested adjacency rule
        adjacencies = neighbors(df, adjacency)
        graph = cls.from_networkx(networkx.Graph(adjacencies))

        graph.geometry = df.geometry

        graph.issue_warnings()

        # Add "exterior" perimeters to the boundary nodes
        add_boundary_perimeters(graph, df.geometry)

        # Add area data to the nodes
        areas = df.geometry.area.to_dict()
        for node, data in graph.nodes():
            data["area"] = areas[node]

        graph.add_data(df, columns=cols_to_add)

        if crs_override is not None:
            df.set_crs(crs_override, inplace=True)

        if df.crs is None:
            warnings.warn(
                "GeoDataFrame has no CRS. Did you forget to set it? "
                "If you're sure this is correct, you can ignore this warning. "
                "Otherwise, please set the CRS using the `crs_override` parameter. "
                "Attempting to proceed without a CRS."
            )
            graph.attrs["crs"] = None
        else:
            graph.attrs["crs"] = df.crs.to_json()

        return graph

    def lookup(self, node: int, field: Any) -> Any:
        """
        Lookup a node/field attribute.

        :param node: Index of the node to look up.
        :type node: int
        :param field: Field to look up.
        :type field: Any

        :returns: The value of the attribute `field` at `node`.
        :rtype: Any
        """
        return self.nodes[node][field]

    def __getitem__(self, node_idx: int) -> dict:
        """
        A slight modification to return the dict of node attributes stored at a node
        rather than the tuple (node_index, node_data_dict) to make working with the
        graph more intuitive.

        :param node_idx: Index of the node to look up.
        :type node_idx: int
        """
        return self.get_node_data(node_idx)[1]

    # NOT DONE
    def add_data(
        self, df: pd.DataFrame, columns: Optional[Iterable[str]] = None
    ) -> None:
        """
        Add columns of a DataFrame to a graph as node attributes
        by matching the DataFrame's index to node ids.

        :param df: Dataframe containing given columns.
        :type df: :class:`pandas.DataFrame`
        :param columns: List of dataframe column names to add. Default is None.
        :type columns: Optional[Iterable[str]], optional

        :returns: None
        """

        if columns is None:
            columns = list(df.columns)

        print(f"In add_data: {columns}")
        self.join(df, columns=columns)

        if "data" in self.attrs:
            self.data[columns] = df[columns]  # type: ignore
        else:
            self.data = df[columns]

    def join(
        self,
        dataframe: pd.DataFrame,
        columns: Optional[List[str]] = None,
        left_index: Optional[str] = None,
        right_index: Optional[str] = None,
    ) -> None:
        """
        Add data from a dataframe to the graph, matching nodes to rows when
        the node's `left_index` attribute equals the row's `right_index` value.
        Unlike the `add_data` method, this method does not add the dataframe
        as a graph attribute nor does it modify the 'data' attribute.

        :param dataframe: DataFrame.
        :type dataframe: :class:`pandas.DataFrame`
        :columns: The columns whose data you wish to add to the graph.
            If not provided, all columns are added. Default is None.
        :type columns: Optional[List[str]], optional
        :left_index: The node attribute used to match nodes to rows.
            If not provided, node IDs are used. Default is None.
        :type left_index: Optional[str], optional
        :right_index: The DataFrame column name to use to match rows
            to nodes. If not provided, the DataFrame's index is used. Default is None.
        :type right_index: Optional[str], optional

        :returns: None
        """
        if right_index is not None:
            df = dataframe.set_index(right_index)
        else:
            df = dataframe

        if columns is not None:
            df = df[columns]

        column_dictionaries = df.to_dict()

        print(column_dictionaries)
        print(self.nodes())

        if left_index is not None:
            ids_to_index = {node: data[left_index] for node, data in self.nodes()}
        else:
            # When the left_index is node ID, the matching is just
            # a redundant {node: node} dictionary
            ids_to_index = dict(zip(self.node_indices(), self.node_indices()))

        node_attributes = {
            node_id: {
                column: values[index] for column, values in column_dictionaries.items()
            }
            for node_id, index in ids_to_index.items()
        }

        for node_id, attributes in node_attributes.items():
            self.get_node_data(node_id)[1].update(attributes)
            print(self.get_node_data(node_id))

    @property
    def connected_components(self) -> list:
        """
        :returns: The set of connected components.
        :rtype: Set
        """
        return rustworkx.connected_components(self)

    def warn_for_disconnected_components(self) -> None:
        """
        :returns: None

        :raises: UserWarning if the graph has any disconnected components.
        """
        components = self.connected_components
        if len(components) > 1:
            warnings.warn(
                f"Found disconnected components. Number of components: {len(components)}. "
                f"Sizes of components: {[len(c) for c in components]}."
            )

    def issue_warnings(self) -> None:
        """
        :returns: None

        :raises: UserWarning if the graph has any red flags (right now, only islands).
        """
        self.warn_for_disconnected_components()


# Done
def add_surcharges(
    graph,
    region_surcharge: dict = None,
):
    if not isinstance(graph, Graph) and not isinstance(graph, rustworkx.PyGraph):
        raise TypeError(f"Unsupported Graph object with type {type(graph)}")

    if region_surcharge is None:
        for edge in graph.edge_list():
            edge_data = graph.get_edge_data(*edge)
            edge_data["surcharge"] = edge_data.get("surcharge", 0.0)
            edge_data["priority"] = edge_data.get("priority", 0)

        return graph

    for edge in graph.edges():
        edge_data = graph.get_edge_data(*edge)
        edge_data["surcharge"] = edge_data.get("surcharge", 0.0)
        edge_data["priority"] = edge_data.get("priority", 0)

        node_0_data = graph.get_node_data(edge[0])
        node_1_data = graph.get_node_data(edge[1])

        for key, value in region_surcharge.items():
            if (
                node_0_data.get(key, None) != node_1_data.get(key, None)
                or node_0_data.get(key, None) is None
                or node_1_data.get(key, None) is None
            ):
                edge_data["surcharge"] += value

            else:
                # Increase the priority for each edge that is in the same region
                # this makes them less likely to be cut
                if (
                    node_0_data.get(key, None) == node_1_data.get(key, None)
                    and node_0_data.get(key, None) is not None
                ):
                    edge_data["priority"] += 1

    return graph


# Done
def add_boundary_perimeters(graph: Graph, geometries: pd.Series) -> None:
    """
    Add shared perimeter between nodes and the total geometry boundary.

    :param graph: NetworkX graph
    :type graph: :class:`Graph`
    :param geometries: :class:`geopandas.GeoSeries` containing geometry information.
    :type geometries: :class:`pandas.Series`

    :returns: The updated graph.
    :rtype: Graph
    """
    from shapely.ops import unary_union
    from shapely.prepared import prep

    prepared_boundary = prep(unary_union(geometries).boundary)

    is_boundary_node_list = geometries.boundary.apply(prepared_boundary.intersects)

    for node, node_data in graph.nodes():
        node_data["boundary_node"] = bool(is_boundary_node_list[node])

        if is_boundary_node_list[node]:
            total_perimeter = geometries[node].boundary.length
            shared_perimeter = sum(
                value["shared_perim"] for _, value in graph.adj(node).items()
            )
            boundary_perimeter = total_perimeter - shared_perimeter
            node_data["boundary_perim"] = boundary_perimeter


# Done
def check_dataframe(df: pd.DataFrame) -> None:
    """
    :returns: None

    :raises: UserWarning if the dataframe has any NA values.
    """
    for column in df.columns:
        if sum(df[column].isna()) > 0:
            warnings.warn("NA values found in column {}!".format(column))


# Done
def remove_geometries(data: networkx.Graph) -> None:
    """
    Remove geometry attributes from NetworkX adjacency data object,
    because they are not serializable. Mutates the ``data`` object.

    Does nothing if no geometry attributes are found.

    :param data: an adjacency data object (returned by
        :func:`networkx.readwrite.json_graph.adjacency_data`)
    :type data: networkx.Graph

    :returns: None
    """
    for node in data["nodes"]:
        bad_keys = []
        for key in node:
            # having a ``__geo_interface__``` property identifies the object
            # as being a ``shapely`` geometry object
            if hasattr(node[key], "__geo_interface__"):
                bad_keys.append(key)
        for key in bad_keys:
            del node[key]


# Done
def convert_geometries_to_geojson(data: networkx.Graph) -> None:
    """
    Convert geometry attributes in a NetworkX adjacency data object
    to GeoJSON, so that they can be serialized. Mutates the ``data`` object.

    Does nothing if no geometry attributes are found.

    :param data: an adjacency data object (returned by
        :func:`networkx.readwrite.json_graph.adjacency_data`)
    :type data: networkx.Graph

    :returns: None
    """
    for node in data["nodes"]:
        for key in node:
            # having a ``__geo_interface__``` property identifies the object
            # as being a ``shapely`` geometry object
            if hasattr(node[key], "__geo_interface__"):
                # The ``__geo_interface__`` property is essentially GeoJSON.
                # This is what :func:`geopandas.GeoSeries.to_json` uses under
                # the hood.
                node[key] = node[key].__geo_interface__


class FrozenGraph:
    """
    Represents an immutable graph to be partitioned. It is based on Rustworkx's PyGraph.

    This speeds up chain runs and prevents having to deal with cache invalidation issues.
    This class behaves slightly differently than :class:`Graph` or :class:`networkx.Graph`.

    Not intended to be a part of the public API.

    :ivar graph: The underlying graph.
    :type graph: rx.PyGraph
    :ivar size: The number of nodes in the graph.
    :type size: int

    Note
    ----
    The class uses `__slots__` for improved memory efficiency.
    """

    __slots__ = ["graph", "size"]

    def __init__(self, graph: Graph) -> None:
        """
        Initialize a FrozenGraph from a Graph.

        :param graph: The mutable Graph to be converted into an immutable graph
        :type graph: Graph

        :returns: None
        """
        self.graph = graph.copy()  # Make a copy to ensure immutability
        self.size = len(self.graph.nodes())

    def __getattribute__(self, __name: str) -> Any:
        try:
            return object.__getattribute__(self, __name)
        except AttributeError:
            if __name in [
                "add_node",
                "add_nodes_from",
                "add_edge",
                "add_edges_from",
                "remove_node",
                "remove_nodes_from",
                "remove_edge",
                "remove_edges_from",
            ]:
                raise RuntimeError("This method is disabled on a frozen graph.")
            return object.__getattribute__(self.graph, __name)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, __name: str) -> Any:
        return self.graph[__name]

    def __iter__(self) -> Iterable[Any]:
        yield from range(self.size)

    @functools.lru_cache(16384)
    def neighbors(self, n: Any) -> Tuple[Any, ...]:
        return tuple(self.graph.neighbors(n))

    @functools.cached_property
    def node_indices(self) -> rustworkx.NodeIndices:
        return self.graph.node_indices

    @functools.cached_property
    def edge_indices(self) -> rustworkx.EdgeIndices:
        return self.graph.edge_indices

    @functools.lru_cache(16384)
    def degree(self, n: Any) -> int:
        return self.graph.degree(n)

    @functools.lru_cache(65536)
    def lookup(self, node: Any, field: str) -> Any:
        return self.graph[node][field]

    # CHECK THIS
    def subgraph(self, nodes: Iterable[Any]) -> "FrozenGraph":
        return FrozenGraph(self.graph.subgraph(nodes))
