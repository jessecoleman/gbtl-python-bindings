from collections import OrderedDict
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

def algorithm(target, algorithm, **containers):

    return get_function(
            target      = "algorithms",
            function    = algorithm,
            args        = [target],
            containers  = containers
    )

def apply(op, const, accum, replace_flag, **containers):

    args = []
    kwargs [("apply_op", op)]

    if const is not None:
        kwargs.append(("bound_const", const))

    if accum is not None:
        kwargs.append(("accum_binaryop", str(accum)))

    else:
        args.append("no_accum")

    return get_function(
            target      = "apply", 
            function    = "apply",
            args        = args, 
            kwargs      = kwargs, 
            containers  = container
    )(replace_flag=replace_flag)
    
def semiring(operator, semiring, accum, replace_flag, **containers):

    args = [operator]

    add_binop, add_idnty, mult_binop = semiring
    kwargs = [
        ("add_binaryop", add_binop),
        ("add_identity", add_idnty),
        ("mult_binaryop", mult_binop),
    ]

    # set default min identity
    if add_idnty == "MinIdentity":
        args.append("min_identity")

    # set default accumulate operator
    if accum != "NoAccumulate":
        kwargs.append(("accum_binaryop", str(accum)))
    else:
        args.append("no_accum")

    get_function(
            target      = "operators",
            function    = operator,
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

