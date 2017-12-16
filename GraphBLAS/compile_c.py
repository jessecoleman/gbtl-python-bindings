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
    args = OrderedDict([
        ("module", hashlib.sha1("nomask".encode("utf-8")).hexdigest())
    ])
    return get_module("nomask", args).NoMask()

def get_utilities():
    args = OrderedDict([
        ("module", hashlib.sha1("utilities".encode("utf-8")).hexdigest())
    ])
    return get_module("utilities", args)

def get_container(dtype):
    args = OrderedDict([("dtype", types[dtype])])
    args["module"] = hashlib.sha1(str(args).encode("utf-8")).hexdigest()
    return get_module("containers", args)

def get_algorithm(algorithm, *type):
    args = OrderedDict([
        (chr(ord('a')+i) + "type", types[t]) for i, t in enumerate(type)
    ])

    args[algorithm] = 1
    args["module"] = hashlib.sha1(str(args).encode("utf-8")).hexdigest()
    return get_module("algorithms", args)

def get_apply(op, const, atype, ctype, mtype, accum):

    args = OrderedDict([
            ("atype", types[atype]),
            ("ctype", types[ctype]),
            ("mtype", types[mtype[0]]),
            (mtype[1], 1),
            ("apply_op", op),
    ])

    # if converting binary op to unary op
    if const is not None: 
        args["bound_const"] = const

    # set default accumulate operator 
    if accum is not None:
        args["accum_binaryop"] = str(accum)
    else: 
        args["no_accum"] = 1

    # generate unique module name from compiler parameters
    args["module"] = hashlib.sha1(str(args).encode("utf-8")).hexdigest()
    return get_module("apply", args)

def get_semiring(add_binaryop, add_identity, mult_binaryop, 
                 atype, btype, ctype, mtype, accum):

    args = OrderedDict([
            ("atype", types[atype]),
            ("btype", types[btype]),
            ("ctype", types[ctype]),
            ("mtype", types[mtype[0]]),
            (mtype[1], 1),
            ("add_binaryop", add_binaryop),
            ("add_identity", add_identity),
            ("mult_binaryop", mult_binaryop)
    ])

    # set default accumulate operator 
    if accum is not None:
        args["accum_binaryop"] = str(accum)
    else: 
        args["no_accum"] = 1

    # set default min identity
    if add_identity == "MinIdentity":
        args["min_identity"] = 1 

    # generate unique module name from macro parameters
    args["module"] = hashlib.sha1(str(args).encode("utf-8")).hexdigest()
    return get_module("operators", args)

# dictionary of GraphBLAS modules
gb = dict()

def get_module(target, args):

    module = args["module"]
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
            "-MD", "-MP", "-MF",
            "{dir}/.deps/binding.Tpo".format(dir=C_CODE),
            "{dir}/binding_{target}.cpp".format(dir=C_CODE, target=target),
            "-o", "{dir}/{mod}{pyext}".format(
                dir=MODULES, 
                mod=module, 
                pyext=PYEXT
            ),
            *("-D{arg}={val}".format(
                arg=str(arg).upper(), val=str(val)) 
                for arg, val in args.items()
            )
    ]
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
        subprocess.check_output(["python3", "-m", "pybind11", "--includes"])
        .decode("ascii").strip()
)

# get file extension for modules
PYEXT       = (
        subprocess.check_output(["python3-config", "--extension-suffix"])
        .decode("ascii").strip()
)

