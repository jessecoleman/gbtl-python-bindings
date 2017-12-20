from GraphBLAS import c_functions as c
from GraphBLAS import Matrix, Vector

def bfs(graph, wavefront, parent_list=None):
    if not isinstance(wavefront, Vector):
        w = [0] * graph.shape[1]
        w[wavefront] = 1
        wavefront = Vector(w)

    if parent_list is None:
        parent_list = Vector([0] * graph.shape[1])

    c.algorithm(
            "bfs_variants",
            "bfs",
            graph       = graph,
            wavefront   = wavefront,
            parent_list = parent_list
    )()
    return parent_list

def bfs_batch(graph, wavefronts, parent_list=None):
    # TODO
    #if not isinstance(wavefront, Matrix):
    #    w = [0] * graph.shape[1]
    #    w[wavefront] = 1
    #    wavefront = Vector(w)

    if parent_list is None:
        parent_list = Matrix(shape=graph.shape, dtype=graph.dtype)

    c.algorithm(
            "bfs_variants",
            "bfs_batch",
            graph       = graph,
            wavefronts  = wavefronts,
            parent_list = parent_list
    )
    return parent_list

def maxflow(capacity, source, sink):
    pass

def vertex_in_degree(graph, vid):
    return c.algorithm(
            "metrics", 
            "vertex_in_degree", 
            graph=graph
    )(vid=vid)

def vertex_out_degree(graph, vid):
    return c.algorithm(
            "metrics", 
            "vertex_out_degree", 
            graph=graph
    )(vid=vid)

def vertex_degree(graph, vid):

    return c.algorithm(
            "metrics", 
            "vertex_degree", 
            graph=graph
    )(vid=vid)

def graph_distance(graph, sid, result=None):

    if result is None:
        result = Vector(shape=graph.shape[0], dtype=graph.dtype)

    return c.algorithm(
            "metrics", 
            "graph_distance", 
            graph=graph,
            result=result,
    )(sid=sid)

def graph_distance_matrix(graph, result=None):

    if result is None:
        result = Matrix(shape=graph.shape, dtype=graph.dtype)

    return c.algorithm(
            "metrics", 
            "graph_distance", 
            graph=graph,
            result=result,
    )()

def vertex_eccentricity(graph, vid):

    return c.algorithm(
            "metrics",
            "vertex_eccentricity",
            graph=graph
    )(vid=vid)

def graph_radius(graph):

    return c.algorithm(
            "metrics",
            "graph_radius",
            graph=graph
    )()

def graph_diameter(graph):

    return c.algorithm(
            "metrics",
            "graph_diameter",
            graph=graph
    )()

def closeness_centrality(graph, vid):
    pass

def get_vertex_IDs(independent_set):
    pass

def mis(graph, independent_set, seed):
    pass

def mst(graph, parents):
    pass

def sssp(matrix, paths):
    a = c.get_algorithm("sssp", matrix.dtype, paths.dtype)
    return a.sssp(matrix.mat, paths.vec)

def triangle_count(l_matrix, u_matrix):
    a = c.get_algorithm("tricount", l_matrix.dtype, u_matrix.dtype)
    return a.triangle_count_newGBTL(l_matrix.mat, u_matrix.mat)

