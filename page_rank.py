from GraphBLAS import *
from GraphBLAS.operators import *


def page_rank(
        graph: Matrix, 
        page_rank: Vector, 
        damping_factor=0.85, 
        threshold=1.e-5, 
        max_iters = None):

        rows, cols = graph.shape

        if rows != cols or page_rank.shape[0] != rows:
            raise Error()

        m = Matrix(shape=graph.shape, dtype=float)

        m[None] = apply(Identity, graph)

        # Normalize the edge weights of the graph by the vertices out-degree
        utilities.normalize_rows(m)

        times_damping_factor = UnaryOp("Times", damping_factor)
        # scale the normalized edge weights by the damping factor
        m[None] = apply(times_damping_factor, m)

        add_scaled_teleport = UnaryOp("Plus", (1.0 - damping_factor) / rows)

        page_rank[:] = 1.0 / rows

        new_rank = Vector(shape=page_rank.shape, dtype=m.dtype)
        delta = Vector(shape=page_rank.shape, dtype=m.dtype)

        i = 0
        while i != max_iters:
        #for i in range(0, max_iters):

            # Compute the new rank: [1 x M][M x N] = [1 x N]
            with Accumulator("Second"), ArithmeticSemiring:
                new_rank[None] += page_rank @ m

            # [1 x M][M x 1] = [1 x 1] = always (1 - damping_factor)
            # rank*(m + scaling_mat*teleport): [1 x 1][1 x M] + [1 x N] = [1 x M]
            new_rank[None] = apply(add_scaled_teleport, new_rank)

            # Test for convergence - compute squared error
            # @todo should be mean squared error. (divide r2/N)
            squared_error = 0.0

            with BinaryOp("Minus"):
                delta[None] = page_rank + new_rank

            with BinaryOp("Times"):
                delta[None] = delta * delta

            squared_error = reduce(PlusMonoid, delta).eval(0.0)

            page_rank[:] = new_rank
            
            # check mean-squared error
            if (squared_error / rows) < threshold:
                return page_rank

        # for any elements missing from page rank vector we need to set
        # to scaled teleport.
        new_rank[:] = (1.0 - damping_factor) / rows

        with BinaryOp("Plus"):
            page_rank[~page_rank] = page_rank + new_rank


def page_rank_naive(
        graph: Matrix, 
        page_rank: Vector, 
        damping_factor=0.85, 
        threshold=1.e-5, 
        max_iters = None):

        rows, cols = graph.shape

        if rows != cols or page_rank.shape[0] != rows:
            raise Error()

        m = Matrix(shape=graph.shape, dtype=float)

        m = apply(Identity, graph)

        # Normalize the edge weights of the graph by the vertices out-degree
        utilities.normalize_rows(m)

        times_damping_factor = UnaryOp("Times", damping_factor)
        # scale the normalized edge weights by the damping factor
        m = apply(times_damping_factor, m)

        add_scaled_teleport = UnaryOp("Plus", (1.0 - damping_factor) / rows)

        page_rank[:] = 1.0 / rows

        new_rank = Vector(shape=page_rank.shape, dtype=m.dtype)
        delta = Vector(shape=page_rank.shape, dtype=m.dtype)

        i = 0
        while i != max_iters:
        #for i in range(0, max_iters):

            # Compute the new rank: [1 x M][M x N] = [1 x N]
            with Accumulator("Second"), ArithmeticSemiring:
                new_rank[None] += page_rank @ m

            # [1 x M][M x 1] = [1 x 1] = always (1 - damping_factor)
            # rank*(m + scaling_mat*teleport): [1 x 1][1 x M] + [1 x N] = [1 x M]
            new_rank = apply(add_scaled_teleport, new_rank)

            # Test for convergence - compute squared error
            # @todo should be mean squared error. (divide r2/N)
            squared_error = 0.0

            with BinaryOp("Minus"):
                delta = page_rank + new_rank

            with BinaryOp("Times"):
                delta = delta * delta

            squared_error = reduce(PlusMonoid, delta).eval(0.0)

            page_rank[:] = new_rank
            
            # check mean-squared error
            if (squared_error / rows) < threshold:
                return page_rank

        # for any elements missing from page rank vector we need to set
        # to scaled teleport.
        new_rank[:] = (1.0 - damping_factor) / rows

        with BinaryOp("Plus"):
            page_rank[~page_rank] = page_rank + new_rank



if __name__=="__main__":
    NUM_NODES = 12;
    i = [
        0, 0, 0, 0,
        1, 1, 1,
        2, 2, 2,
        3, 3, 3, 3,
        4, 4, 4, 4,
        5, 5,
        6, 6, 6,
        7, 7, 7, 7,
        8, 8, 8, 8,
        9, 9, 9,
        10,10,10,10,
        11,11]

    j = [
        1, 5, 6, 9,
        0, 2, 4,
        1, 3, 4,
        2, 7, 8, 10,
        1, 2, 6, 7,
        0, 9,
        0, 4, 9,
        3, 4, 8, 10,
        3, 7, 10, 11,
        0, 5, 6,
        3, 7, 8, 11,
        8, 10]

    v = [1.0] * len(i)
    
    m1 = Matrix((v, (i, j)), shape=(NUM_NODES, NUM_NODES))
    #print(m1)

    rank = Vector(shape=NUM_NODES, dtype=float)
    
    page_rank(m1, rank)

    print(rank)
    rank = Vector(shape=NUM_NODES, dtype=float)
        
    algorithms.page_rank(m1, rank)

