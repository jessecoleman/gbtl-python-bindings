import attr
from collections import OrderedDict
import inspect
from functools import partial
import numpy as np
from . import c_modules as c_mod

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

def container(dtype):

    return get_function(
            target = "containers", 
            kwargs = [("dtype", types[dtype])]
    )

def algorithm(target, algorithm=None, **containers):

    if algorithm is None:
        algorithm = target

    return get_function(
            target      = "algorithms",
            function    = algorithm,
            args        = [target],
            containers  = containers
    )

def operator(operator, function, accum, replace_flag, **containers):

    ops = attr.asdict(operator)

    # set bound constant if apply binary_op
    if ops.get("bound_const", False) is None:
        del ops["bound_const"]

    args = [type(operator).__name__, function]
    kwargs = list(ops.items())

    # set default min identity
    if ops.get("add_identity", None) == "MinIdentity":
        args.append("min_identity")

    # set default accumulate operator
    if accum != "NoAccumulate":
        kwargs.append(("accum_binaryop", str(accum)))
    else:
        args.append("no_accum")

    get_function(
            target      = "operators",
            function    = function,
            args        = args,
            kwargs      = kwargs,
            containers  = containers
    )(replace_flag=replace_flag)

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

import inspect
def get_function(target, function=None, args=None, kwargs=None, containers=None):

    if args is None: args = []
    if kwargs is None: kwargs = []

    if isinstance(containers, dict):
        for i, c in containers.items():
            args.append(i + "_" + type(c).__name__)
            kwargs.append((i + "_type", types[c.dtype]))

    module = c_mod.cache[target, args, kwargs]

    # partially apply function with container arguments
    if function is not None:
        return partial(
                getattr(module, function), 
                **{i: c.container for i, c in containers.items()}
        )

    else: return module

def typecheck(function):
    def _f(*args):
        args = list(args)
        print(inspect.getfullargspec(function)[0])
        for i, arg in enumerate(inspect.getfullargspec(function)[0]):
            while not isinstance(args[i], function.__annotations__[arg]):
                try:
                    args[i] = args[i].evaluated
                except:
                    raise TypeError("{} is not of type {}".format(args[i], arg))
        return function(*args)
    _f.__doc__ = function.__doc__
    return _f
    
