from collections import OrderedDict
import hashlib
import importlib
import inspect
import numpy as np
import os
import subprocess
import sys

# mapping from python/numpy types to c types
# ordered by typecasting heirarchy
types = OrderedDict([
        (None, ""),
        (bool, "bool"),
        (np.bool_, "bool"),
        (np.int8, "int8_t"),
        (np.uint8, "uint8_t"),
        (np.int16, "int16_t"),
        (np.uint16, "uint16_t"),
        (np.int32, "int32_t"),
        (np.uint32, "uint32_t"),
        (int, "int64_t"),
        (np.int64, "int64_t"),
        (np.uint64, "uint64_t"),
        (np.float32, "float"),
        (float, "double"),
        (np.float64, "double")
])

def get_type(container):
    # if a is a numpy/scipy array
    try: return container.dtype.type
    # if a is an N-D list/array
    except AttributeError:
        # drill down to data in container
        while type(container) not in types: 
            container = container[0]
        return type(container)

# upcast ctype to largest of atype and btype
def upcast(atype, btype):
    py_types = list(types.keys())
    return list(types.items())[max(
            py_types.index(atype), 
            py_types.index(btype)
    )][0]

def no_mask():
    return get_module("nomask", []).NoMask()

def get_utilities():
    args = [("utilities", 1)]
    return get_module("utilities", args)

def get_container(dtype):
    args = [("dtype", types[dtype])]
    return get_module("containers", args)

def alpha(i): return chr(ord('a') + i)

def type_params(*containers):

    params = []
    for i, container in enumerate(containers):
        params.append((alpha(i) + "type", types[container.dtype]))
        params.append((alpha(i) + "_" + type(container).__name__, 1))

    return params

def get_algorithm(algorithm, *containers):
    args = type_params(*containers)
    args.append((algorithm, 1))
    return get_module("algorithms", args)

def get_apply(op, const, args, accum):

    args.append(("apply_op", op))

    if const is not None: 
        args.append(("bound_const", const))

    if accum is not None:
        args.append(("accum_binaryop", str(accum)))

    else: 
        args.append(("no_accum", 1))

    return get_module("apply", args)

def get_semiring(add_binop, add_idnty, mult_binop, op, accum, args):

    args.append(("add_binaryop", add_binop))
    args.append(("add_identity", add_idnty))
    args.append(("mult_binaryop", mult_binop))
    args.append((op, 1))

    # set default min identity
    if add_idnty == "MinIdentity":
        args.append(("min_identity", 1))

    # set default accumulate operator 
    if accum is not None:
        args.append(("accum_binaryop", str(accum)))
    else: 
        args.append(("no_accum", 1))

    return get_module("operators", args)

# dictionary of GraphBLAS modules
gb = dict()

def get_module(target, args):

    module = hashlib.sha1(str(args).encode("utf-8")).hexdigest()
    if module not in gb:
        try:
            gb[module] = importlib.import_module(
                    "GraphBLAS.modules.{mod}".format(mod=module)
            )

        except ImportError:
            gb[module] = build_module(target, module, args)

    return gb[module]

def build_module(target, module, args):

    if not os.path.exists(MODULES):
        os.makedirs(MODULES)
    
    cmd = [
            CXX,
            LANG,
            *OPTS.split(),
            *FLAGS.split(),
            *PYBIND.split(),
            #PICKY, DEBUG,
            "-I{dir}".format(dir=C_CODE),
            "-I{gb_source}".format(gb_source=GB_SOURCE),
            "-MT", "graphblas{pyext}".format(pyext=PYEXT),
            *"-MD -MP -MF".split(),
            "{dir}/.deps/binding.Tpo".format(dir=C_CODE),
            "{dir}/binding_{target}.cpp".format(dir=C_CODE, target=target),
            "-o", "{dir}/{mod}{pyext}".format(
                dir=MODULES, 
                mod=module, 
                pyext=PYEXT
            ),
            "-DMODULE={}".format(module),
            *("-D{arg}={val}".format(
                arg=str(arg).upper(), val=str(val))
                for arg, val in args
            )
    ]
    print(cmd)
    subprocess.call(cmd, cwd=C_CODE)

    return importlib.import_module(
            "GraphBLAS.modules.{mod}".format(mod=module)
    )

# compiler flags
CXX         = "g++"
LANG	    = "-std=c++14"
OPTS	    = "-O3 -march=native -DNDEBUG -shared -fPIC -fvisibility=hidden"
PICKY	    = "-Wall"
DEBUG 	    = "-g"
FLAGS       = "-DHAVE_CONFIG_H -DGB_USE_SEQUENTIAL"
GB_SOURCE   = "/home/jessecoleman/graphpack/gbtl/src"

# project directories
CWD         = inspect.getfile(inspect.currentframe()).rsplit("/", 1)[0]
MODULES     = os.path.abspath("{cwd}/modules".format(cwd=CWD))
C_CODE      = os.path.abspath("{cwd}/c_code".format(cwd=CWD))
sys.path.append(CWD)

# get environment variables
PYBIND      = (
        subprocess.check_output("python3 -m pybind11 --includes".split())
        .decode("ascii").strip()
)

# get file extension for modules
PYEXT       = (
        subprocess.check_output("python3-config --extension-suffix".split())
        .decode("ascii").strip()
)

