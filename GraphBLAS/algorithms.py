import sys
from GraphBLAS import c_functions as c
from GraphBLAS import Matrix, Vector

@c.type_check
def bfs(graph: Matrix, wavefront: Vector, parent_list=None):
    if not isinstance(wavefront, Vector):
        print("creating wavefront")
        w = [0] * graph.shape[1]
        w[wavefront] = 1
        wavefront = Vector(w)

    if parent_list is None:
        print("creating parent_list")
        parent_list = Vector([0] * graph.shape[1])

    c.algorithm(
            algorithm   = "bfs",
            graph       = graph,
            wavefront   = wavefront,
            parent_list = parent_list
    )

    #return parent_list

@c.type_check
def bfs_batch(graph: Matrix, wavefronts: Matrix, parent_list=None):
    # TODO
    #if not isinstance(wavefront, Matrix):
    #    w = [0] * graph.shape[1]
    #    w[wavefront] = 1
    #    wavefront = Vector(w)

    if parent_list is None:
        parent_list = Matrix(shape=graph.shape, dtype=graph.dtype)

    c.algorithm(
            algorithm   = "bfs_batch",
            graph       = graph,
            wavefronts  = wavefronts,
            parent_list = parent_list
    )

    return parent_list

def bfs_level_masked_v2(graph, wavefront, levels):


@c.type_check
def maxflow(capacity: Matrix, source: int, sink: int):

    return c.algorithm(
            algorithm   = "maxflow",
            capacity    = capacity
    )(
            source      = source,
            sink        = sink
    )

@c.type_check
def vertex_in_degree(graph: Matrix, vid: int):

    return c.algorithm(
            algorithm   = "vertex_in_degree", 
            graph       = graph
    )(
            vid         = vid
    )

@c.type_check
def vertex_out_degree(graph: Matrix, vid: int):
    return c.algorithm(
            algorithm   = "vertex_out_degree", 
            graph       = graph
    )(      
            vid         = vid
    )

@c.type_check
def vertex_degree(graph: Matrix, vid: int):

    return c.algorithm(
            algorithm   = "vertex_degree", 
            graph       = graph
    )(
            vid         = vid
    )

@c.type_check
def graph_distance(graph: Matrix, sid: int, result: Vector=None):

    if result is None:
        result = Vector(shape=graph.shape[0], dtype=graph.dtype)

    return c.algorithm(
            algorithm   = "graph_distance", 
            graph       = graph,
            result      = result,
    )(
            sid         = sid
    )

@c.type_check
def graph_distance_matrix(graph: Matrix, result: Matrix = None):

    if result is None:
        result = Matrix(shape=graph.shape, dtype=graph.dtype)

    return c.algorithm(
            algorithm   = "graph_distance", 
            graph       = graph,
            result      = result,
    )

@c.type_check
def vertex_eccentricity(graph: Matrix, vid: int):

    return c.algorithm(
            algorithm   = "vertex_eccentricity",
            graph       = graph
    )(
            vid         = vid
    )

@c.type_check
def graph_radius(graph: Matrix):

    return c.algorithm(
            algorithm   = "graph_radius",
            graph       = graph
    )

@c.type_check
def graph_diameter(graph: Matrix):

    return c.algorithm(
            algorithm   = "graph_diameter",
            graph       = graph
    )

@c.type_check
def closeness_centrality(graph: Matrix, vid: int):

    return c.algorithm(
            algorithm   = "closeness_centrality",
            graph       = graph
    )(
            vid         = vid
    )

@c.type_check
def get_vertex_IDs(independent_set: Matrix):

    return c.algorithm(
            algorithm       = "get_vertex_IDs",
            independent_set = independent_set
    )

@c.type_check
def mis(graph: Matrix, independent_set: Vector, seed: int = 0):

    return c.algorithm(
            algorithm       = "mis",
            graph           = graph,
            independent_set = independent_set
    )(
            seed            = seed        
    )

@c.type_check
def mst(graph: Matrix, parents: Vector):

    return c.algorithm(
            algorithm   = "mst",
            graph       = graph,
            parents     = parents
    )

@c.type_check
def page_rank(
        graph: Matrix, 
        page_rank: Vector,
        damping_factor=0.85,
        threshold=1.e-5,
        max_iters=None):

    if max_iters is None:

        c.algorithm(
                algorithm       = "page_rank",
                graph           = graph,
                page_rank       = page_rank,
                damping_factor  = damping_factor,
                threshold       = threshold,
        )

    else:

        c.algorithm(
                algorithm       = "page_rank",
                graph           = graph,
                page_rank       = page_rank,
                damping_factor  = damping_factor,
                threshold       = threshold,
                max_iters       = max_iters
        )

    print("page_rank:", page_rank)

    return page_rank
 

@c.type_check
def sssp(matrix: Matrix, paths: Vector):

    return c.algorithm(
            algorithm   = "sssp", 
            matrix      = matrix, 
            paths       = paths
    )

@c.type_check
def triangle_count(graph: Matrix):
    return c.algorithm(
            algorithm   = "triangle_count",
            graph       = graph
    )

@c.type_check
def triangle_count_newGBTL(l_matrix: Matrix, u_matrix: Matrix):

    return c.algorithm(
            algorithm   = "triangle_count_newGBTL",
            l_matrix    = l_matrix,
            u_matix     = u_matrix
    )

