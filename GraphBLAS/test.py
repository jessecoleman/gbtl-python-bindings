from collections import OrderedDict

def test(**kwargs):
    print(kwargs)

o = OrderedDict([
    ("1", 2),
    ("2", 3)
])

test(**o)
