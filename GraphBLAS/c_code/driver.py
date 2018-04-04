import container

m = container.PyMatrix([0, 1, 0, 1], [0, 0, 1, 1], [1, 2, 3, 4], (2, 2))
n = container.PyMatrix([0, 1, 0, 1], [0, 0, 1, 1], [1, 2, 3, 4], (2, 2))

o = container.PyMatrix([0, 1, 0, 1], [0, 0, 1, 1], [1, 2, 3, 4], (2, 2))

print(m.shape, m.nvals)
        
print(m[0, 0], m[0, 1])
print(m)

#print(container.Pymxm(o, container.NoAccumulate(), container.ArithmeticSemiring(), m, n, True))
