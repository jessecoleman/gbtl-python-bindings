from GraphBLAS import c_functions as c
from GraphBLAS import Matrix, Vector

@c.typecheck
def bfs(graph: Matrix, wavefront: Vector, parent_list=None):
    if not isinstance(wavefront, Vector):
        w = [0] * graph.shape[1]
        w[wavefront] = 1
        wavefront = Vector(w)

    if parent_list is None:
        parent_list = Vector([0] * graph.shape[1])

    c.algorithm(
            target      = "bfs_variants",
            algorithm   = "bfs",
            graph       = graph,
            wavefront   = wavefront,
            parent_list = parent_list
    )()

    return parent_list

@c.typecheck
def bfs_batch(graph: Matrix, wavefronts: Matrix, parent_list=None):
    # TODO
    #if not isinstance(wavefront, Matrix):
    #    w = [0] * graph.shape[1]
    #    w[wavefront] = 1
    #    wavefront = Vector(w)

    if parent_list is None:
        parent_list = Matrix(shape=graph.shape, dtype=graph.dtype)

    c.algorithm(
            target      = "bfs_variants",
            algorithm   = "bfs_batch",
            graph       = graph,
            wavefronts  = wavefronts,
            parent_list = parent_list
    )()

    return parent_list

@c.typecheck
def maxflow(capacity: Matrix, source: int, sink: int):

    return c.algorithm(
            target      = "maxflow",
            algorithm   = "maxflow",
            capacity    = capacity
    )(
            source      = source,
            sink        = sink
    )

@c.typecheck
def vertex_in_degree(graph: Matrix, vid: int):

    return c.algorithm(
            target      = "metrics", 
            algorithm   = "vertex_in_degree", 
            graph       = graph
    )(
            vid         = vid
    )

@c.typecheck
def vertex_out_degree(graph: Matrix, vid: int):
    return c.algorithm(
            target      = "metrics", 
            algorithm   = "vertex_out_degree", 
            graph       = graph
    )(      
            vid         = vid
    )

@c.typecheck
def vertex_degree(graph: Matrix, vid: int):

    return c.algorithm(
            target      = "metrics", 
            algorithm   = "vertex_degree", 
            graph       = graph
    )(
            vid         = vid
    )

@c.typecheck
def graph_distance(graph: Matrix, sid: int, result: Vector=None):

    if result is None:
        result = Vector(shape=graph.shape[0], dtype=graph.dtype)

    return c.algorithm(
            target      = "metrics", 
            algorithm   = "graph_distance", 
            graph       = graph,
            result      = result,
    )(
            sid         = sid
    )

@c.typecheck
def graph_distance_matrix(graph: Matrix, result: Matrix = None):

    if result is None:
        result = Matrix(shape=graph.shape, dtype=graph.dtype)

    return c.algorithm(
            target      = "metrics", 
            algorithm   = "graph_distance", 
            graph       = graph,
            result      = result,
    )()

@c.typecheck
def vertex_eccentricity(graph: Matrix, vid: int):

    return c.algorithm(
            target      = "metrics",
            algorithm   = "vertex_eccentricity",
            graph       = graph
    )(
            vid         = vid
    )

@c.typecheck
def graph_radius(graph: Matrix):

    return c.algorithm(
            target      = "metrics",
            algorithm   = "graph_radius",
            graph       = graph
    )()

@c.typecheck
def graph_diameter(graph: Matrix):

    return c.algorithm(
            target      = "metrics",
            algorithm   = "graph_diameter",
            graph       = graph
    )()

@c.typecheck
def closeness_centrality(graph: Matrix, vid: int):

    return c.algorithm(
            target      = "metrics",
            algorithm   = "closeness_centrality",
            graph       = graph
    )(
            vid         = vid
    )

@c.typecheck
def get_vertex_IDs(independent_set: Matrix):

    return c.algorithm(
            target          = "mis",
            algorithm       = "get_vertex_IDs",
            independent_set = independent_set
    )()

@c.typecheck
def mis(graph: Matrix, independent_set: Vector, seed: int = 0):

    return c.algorithm(
            target          = "mis",
            algorithm       = "mis",
            graph           = graph,
            independent_set = independent_set
    )(
            seed            = seed        
    )

@c.typecheck
def mst(graph: Matrix, parents: Vector):

    return c.algorithm(
            target      = "mst",
            algorithm   = "mst",
            graph       = graph,
            parents     = parents
    )()

@c.typecheck
def sssp(matrix: Matrix, paths: Vector):

    return c.algorithm(
            target      = "sssp", 
            matrix      = matrix, 
            paths       = paths
    )()

@c.typecheck
def triangle_count(l_matrix: Matrix, u_matrix: Matrix):
    return c.algorithm(
            target      = "tricount", 
            algorithm   = "triangle_count",
            l_matrix    = l_matrix, 
            u_matrix    = u_matrix
    )()

@c.typecheck
def triangle_count_newGBTL(l_matrix: Matrix, u_matrix: Matrix):

    return c.algorithm(
            target      = "tricount",
            algorithm   = "triangle_count_newGBTL",
            l_matrix    = l_matrix,
            u_matix     = u_matrix
    )()

