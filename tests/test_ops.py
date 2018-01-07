import GraphBLAS as gb

m = gb.Matrix(([0], ([0], [0])))

print(m)

g = m * m

print(type(g))
