import attr
from collections import OrderedDict
import inspect
import numpy as np
from toolz import curry
from . import c_modules

# mapping from python/numpy types to c types
# ordered by typecasting heirarchy
types = OrderedDict([
        (None,      ""),
        (bool,      "bool"),
        (np.bool_,  "bool"),
        (np.int8,   "int8_t"),
        (np.uint8,  "uint8_t"),
        (np.int16,  "int16_t"),
        (np.uint16, "uint16_t"),
        (np.int32,  "int32_t"),
        (np.uint32, "uint32_t"),
        (int,       "int64_t"),
        (np.int64,  "int64_t"),
        (np.uint64, "uint64_t"),
        (np.float32,"float"),
        (float,     "double"),
        (np.float64,"double")
])


def no_mask():
    return get_function(target = "nomask").NoMask()

def all_indices():
    return get_function(target = "nomask").AllIndices()

def container(dtype):

    return get_function(
            target = "containers", 
            kwargs = {"dtype": types[dtype]}
    )

def bench(algorithm, **containers):

    alg_group = algorithm_dict.get(algorithm, algorithm)

    return get_function(
            target      = "benchmarks",
            function    = "benchmark_" + algorithm,
            args        = [alg_group],
            containers  = containers
    )

def operator(function, operation=None, accum=None, replace_flag=None, **containers):

    args = [function]
    kwargs = {}

    if operation is not None:

        args.append(type(operation).__name__)
        operators = operation.__dict__

        # set default min identity
        if operators.get("add_identity", None) == "MinIdentity":
            args.append("min_identity")

        kwargs.update(operators)

    # set default accumulate operator
    if not accum:
        args.append("no_accum")

    else:
        kwargs["accum_binaryop"] = accum.binary_op

    f = get_function(
            target      = "operators",
            function    = function,
            args        = args,
            kwargs      = kwargs,
            containers  = containers
    )

    if replace_flag is None: return f
    else: return f(replace_flag=replace_flag)

algorithm_dict = {
        "bfs": "bfs",
        "bfs_batch": "bfs",
        "bfs_level": "bfs",
        "bfs_level_masked_v2": "bfs",
        "vertex_in_degree": "metrics",
        "vertex_out_degree": "metrics",
        "vertex_degree": "metrics",
        "graph_distance": "metrics",
        "graph_distance_matrix": "metrics",
        "vertex_eccentricity": "metrics",
        "graph_radius": "metrics",
        "graph_diameter": "metrics",
        "closeness_centrality": "metrics",
        "get_vertex_IDs": "mis",
        "triangle_count": "triangle_count",
        "triangle_count_masked": "triangle_count",
        "triangle_count_flame1": "triangle_count",
        "triangle_count_flame1a": "triangle_count",
        "triangle_count_flame2": "triangle_count",
        "triangle_count_newGBTL": "triangle_count"
}
        
def algorithm(algorithm, **containers):

    alg_group = algorithm_dict.get(algorithm, algorithm)

    return get_function(
            target      = "algorithms",
            function    = algorithm,
            args        = [alg_group],
            containers  = containers
    )

def utilities(function, args=None, kwargs=None, **containers):

    return get_function(
            target      = "utilities",
            function    = function,
            args        = args,
            kwargs      = kwargs,
            containers  = containers
    )

# upcast ctype to largest of atype and btype
def upcast(atype, btype):
    py_types = list(types.keys())
    return list(types.items())[max(
            py_types.index(atype),
            py_types.index(btype)
    )][0]

def get_type(container):
    # if a is a numpy/scipy array
    try: return container.dtype.type
    # if a is an N-D list/array
    except AttributeError:
        # drill down to data in container
        while type(container) not in types:
            container = container[0]
        return type(container)

def get_function(target, function=None, args=None, kwargs=None, containers=None):

    if args is None: args = []
    if kwargs is None: kwargs = {}

    if containers is not None:
        for i, c in containers.items():
            # if c is container
            if hasattr(c, "dtype"):
                kwargs[i + "_type"] = types[c.dtype]
                args.append(i + "_" + type(c).__name__)
            # if c is value
            elif type(c) in types:
                kwargs[i + "_type"] = types[type(c)]
                args.append(i + "_value")
            # if c is index list
            elif type(c).__name__ == "AllIndices":
                args.append("all_" + i)
            else:
                kwargs[i + "_type"] = types[get_type(c)]
                args.append(i)

    module = c_modules.cache[target, args, kwargs]

    # partially apply function with container arguments
    if function is not None:
        return curry(getattr(module, function))(**{
                i: (c.container if hasattr(c, "container") else c) 
                for i, c in containers.items()
        })

    else:
        return module

def type_check(function):
    def _f(*args):
        args = list(args)
        #print(inspect.getfullargspec(function)[0])
        for i, arg in enumerate(inspect.getfullargspec(function)[0]):
            if arg == 'self': continue
        #    print(function.__annotations__[arg])
        #    print(type(function.__annotations__[arg]))
        #    while not isinstance(args[i], function.__annotations__[arg]):
        #        try:
        #            args[i] = args[i].evaluated
        #        except:
        #            raise TypeError("{} is not of type {}".format(args[i], arg))
        return function(*args)
    _f.__doc__ = function.__doc__
    return _f
    
def dim_check(function):
    def _f(operator, A, B, *args):
        #print(args)
        if function.__name__.startswith("eWise") and A.shape != B.shape:
            raise Exception("dimensions of {} and {} must match".format(A, B))
        elif function.__name__ == "mxm" and A.shape[1] != B.shape[0]:
            raise Exception("columns of {} must be the same as rows of {}".format(A, B))
        elif function.__name__ == "mxv" and A.shape[1] != B.shape[0]:
            raise Exception("columns of {} must be the same as length of {}".format(A, B))
        elif function.__name__ == "vxm" and A.shape[0] != B.shape[0]:
            raise Exception("rows of {} must be the same as length of {}".format(A, B))

        return function(operator, A, B, *args)
    _f.__doc__ = function.__doc__
    return _f

