from GraphBLAS import compile_c as c

def bfs_level(matrix, root):
    _, t = c._get_type(matrix)
    a = c.get_algorithm(t, alg="BFS")
    return a.bfs_level(matrix.mat, root.mat)

#def: maxflow(matrix, 
#def: metrics(
#def: mis(
#def: mst(

def sssp(matrix, paths):
    _, t = c._get_type(matrix)
    a = c.get_algorithm(t, alg="SSSP")
    return a.sssp(matrix.mat, paths.vec)

def triangle_count(l_matrix, u_matrix):
    _, t = c._get_type(l_matrix)
    a = c.get_algorithm(t, alg="TRICOUNT")
    return a.triangle_count_newGBTL(l_matrix.mat, u_matrix.mat)


