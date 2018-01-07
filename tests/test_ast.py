from dill.source import getsource
import inspect
from itertools import product
import ast


def s(*slices):
    
    p = product(*(range(*s.indices(10)) for s in slices))

    #print(p)

    for s in p:
        print(s)


    #print(product(range(s.indices((10, 10))) for s in slices))


s(slice(1,2),slice(4,8), slice(0,10,2))

exit()

def func(test, t):
    for a in range(10):
        print(a)

        
t = inspect.getsourcelines(func)
print(node)

print(t)


p = ast.parse(t)

for node in ast.walk(p):
    print(node)
    print(dir(node))
    print("Attributes:", node._attributes)
    print("Fields:", [node._fields[i] for i, f in enumerate(node._fields)])
    #print(dir(node))

