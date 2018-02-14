import GraphBLAS as gb
from GraphBLAS import Matrix, Vector
from GraphBLAS.operators import LogicalSemiring, MinSelect2ndSemiring, Replace
from GraphBLAS import utilities

def index_of_1based(vector):
    i, j, v = [],[],[]
    
    for idx in range(len(vector)):
        i.append(idx)
        j.append(idx)
        v.append(idx+1)

    identity_ramp = gb.Matrix((v,(i,j)))

    with MinSelect2ndSemiring, Replace:
        vector[None] = vector @ identity_ramp

    return vector

def bfs_parent_list(graph, wavefront, parent_list):

    parent_list = index_of_1based(parent_list)

    while wavefront.nvals > 0:

        wavefront = index_of_1based(wavefront)

        with PlusAccumulate, MinSelect1stSemiring, Replace:
            wavefront[~parent_list] = wavefront.T @ graph

        with Accumulate("Times"), LogicalSemiring:
            parent_list[None] += wavefront

    subtract_1 = UnaryOp("Minus", 1)

    parent_list[None] = apply(subtract_1, parent_list)


def bfs_level_masked_v2(graph, wavefront, levels):

    wavefront = Vector(wavefront)

    g_rows, g_cols = graph.shape

    w_size = wavefront.shape[0]

    if g_rows != g_cols or w_size != g_rows:
        raise Error()

    depth = 0
    while wavefront.nvals > 0:
        # Increment the level
        depth += 1

        levels[wavefront][:] = depth

        # Advance the wavefront and mask out nodes already assigned levels
        with LogicalSemiring, Replace:
            wavefront[~levels] = wavefront @ graph

def bfs_level_masked_naive(graph, wavefront, levels):

    wavefront = Vector(wavefront)

    g_rows, g_cols = graph.shape

    w_size = wavefront.shape[0]

    if g_rows != g_cols or w_size != g_rows:
        raise Error()

    depth = 0
    while wavefront.nvals > 0:
        # Increment the level
        depth += 1

        levels[wavefront] = depth

        # Advance the wavefront and mask out nodes already assigned levels
        with LogicalSemiring, Replace:
            wavefront[~levels] = wavefront @ graph

