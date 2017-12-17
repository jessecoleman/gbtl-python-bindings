from GraphBLAS import compile_c as c
from GraphBLAS import Matrix, Vector

def bfs(graph, wavefront, parent_list=None):
    if not isinstance(wavefront, Vector):
        w = [0] * graph.shape[1]
        w[wavefront] = 1
        wavefront = Vector(w)

    if parent_list is None:
        parent_list = Vector([0] * graph.shape[1])

    a = c.get_algorithm(
            "bfs",
            graph,
            wavefront,
            parent_list
    )
    a.bfs(graph.mat, wavefront.vec, parent_list.vec)
    return parent_list

def maxflow(capacity, source, sink):
    pass

def vertex_in_degree(graph, vid):
    pass

def vertex_out_degree(graph, vid):
    pass

def vertex_degree(graph, vid):
    pass

def graph_distance(graph, sid, result):
    pass

def graph_distance_matrix(graph, result):
    pass

def vertex_eccentricity(graph, vid):
    pass

def graph_radius(graph):
    pass

def graph_diameter(graph):
    pass

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

