from functools import partial

def test(a, b, c):
    print(a, b, c)

part = partial(test, a=1, b=2, d=3)

part(b=4)
