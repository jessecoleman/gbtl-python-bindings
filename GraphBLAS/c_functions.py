from functools import partial
from .c_modules import module_cache, types

def no_mask():
    return module_cache["nomask",[],[],[]].NoMask()

def get_container(dtype):
    kwargs = [("dtype", types[dtype])]
    return module_cache["containers", [], kwargs, []]

def algorithm(group, algorithm, **containers):
    args = [group]
    module, f_args = module_cache["algorithms", args, [], containers]
    return partial(getattr(module, algorithm), **f_args)

def apply(op, const, accum, replace, **containers):

    args = []
    kwargs = []
    kwargs.append(("apply_op", op))

    if const is not None:
        kwargs.append(("bound_const", const))

    if accum is not None:
        kwargs.append(("accum_binaryop", str(accum)))

    else:
        args.append("no_accum")

    module, f_args = module_cache["apply", args, kwargs, containers]
    module.apply(
            **f_args,
            replace_flag=replace
    )

def semiring(operator, semiring, accum, replace, **containers):

    add_binop, add_idnty, mult_binop = semiring

    kwargs = [
        ("add_binaryop", add_binop),
        ("add_identity", add_idnty),
        ("mult_binaryop", mult_binop),
    ]

    args = [operator]

    # set default min identity
    if add_idnty == "MinIdentity":
        args.append("min_identity")

    # set default accumulate operator
    if accum != "NoAccumulate":
        kwargs.append(("accum_binaryop", str(accum)))
    else:
        args.append("no_accum")

    module, f_args = module_cache["operators", args, kwargs, containers]
    getattr(module, operator)(
            **f_args,
            replace_flag=replace
    )

def get_utilities(args, type=None):
    #if type is None:
    #    args = type_params(*containers)
    #else:
    #    args = type
    print("getting utilities", args)
    return get_module("utilities", args)

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

