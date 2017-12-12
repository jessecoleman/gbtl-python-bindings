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
            "BFS",
            graph.dtype, 
            wavefront.dtype, 
            parent_list.dtype
    )
    a.bfs(graph.mat, wavefront.vec, parent_list.vec)
    return parent_list

#def: maxflow(matrix, 
#def: metrics(
#def: mis(
#def: mst(

def sssp(matrix, paths):
    t = c.get_type(matrix)
    a = c.get_algorithm(t, alg="SSSP")
    return a.sssp(matrix.mat, paths.vec)

def triangle_count(l_matrix, u_matrix):
    t = c.get_type(l_matrix)
    a = c.get_algorithm(t, alg="TRICOUNT")
    return a.triangle_count_newGBTL(l_matrix.mat, u_matrix.mat)


