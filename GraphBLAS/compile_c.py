import os
import sys
import subprocess
import importlib
import inspect
from collections import OrderedDict
import hashlib
import numpy as np

# dictionary of GraphBLAS modules
_gb = dict()

# mapping from python/numpy types to c types
# ordered by typecasting heirarchy
_types = OrderedDict([
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

# get module directory
_CWD = inspect.getfile(inspect.currentframe()).rsplit("/", 1)[0]
sys.path.append(_CWD)
_MODDIR = os.path.abspath(_CWD + "/lib")

# get environment variables
_PYBIND = (
        subprocess.check_output(["python3", "-m", "pybind11", "--includes"])
        .decode("ascii").strip().split(" ")
)

_PYEXT = (
        subprocess.check_output(["python3-config", "--extension-suffix"])
        .decode("ascii").strip()
)

# upcast ctype to largest of atype and btype
def upcast(atype, btype):
    py_types = list(_types.keys())
    return list(_types.items())[max(
            py_types.index(atype), 
            py_types.index(btype)
    )][0]

def get_container(atype):
    module = "at_" + _types[atype]
    module = hashlib.sha1(module.encode('utf-8')).hexdigest()
    args = {"atype": _types[atype]}
    if _types[atype] == 'bool': args['mask'] = "1"
    else: args["mask"] = "0"
    return _get_module("container", module, **args)

def get_algorithm(algorithm, atype, btype):
    c_types = {"atype": _types[atype]}
    if btype is not None:
        c_types["btype"] = _types[btype]
        c_types["ctype"] = _types[upcast(atype, btype)]
    else: c_types["btype"] = ""

    module  = "at_" + _types[atype]\
            + "bt_" + _types[btype]\
            + "al_" + algorithm
    module = hashlib.sha1(module.encode('utf-8')).hexdigest()

    args = c_types
    args.update({"alg": algorithm})
    return _get_module("algorithm", module, args)

def get_apply(op, const, atype, ctype, accum):
    c_types = {"atype": _types[atype], "ctype": _types[ctype]}
    module  = "at_" + _types[atype]\
            + "ct_" + _types[atype]\
            + "ap_" + op\
            + "const_" + str(const)\
            + "ac_" + str(accum)
    # generate unique module name from compiler parameters
    module = hashlib.sha1(module.encode('utf-8')).hexdigest()

    args = c_types
    args["apply_op"] = op
    # if converting binary op to unary op
    if const is None:
        args["bound_second"] = "0"
    else: 
        args["bound_second"] = "1"
        args["bound_const"] = str(const)
    # set default accumulate operator 
    if accum is None:
        args["accum_binaryop"] = "NoAccumulate"
        args["no_accum"] = "1" 
    else: 
        args["accum_binaryop"] = accum
        args["no_accum"] = "0"
    return _get_module("apply", module, **args)

def get_semiring(semiring, atype, btype, ctype, accum):
    # TODO accept ctype if accumulation is set
    c_types = {
            "atype": _types[atype],
            "btype": _types[btype],
            "ctype": _types[ctype],
    }
    module  = "at_" + _types[atype]\
            + "bt_" + _types[btype]\
            + "ct_" + _types[ctype]\
            + "sr_" + "".join(map(str,semiring))\
            + "ac_" + str(accum)
    # generate unique module name from compiler parameters
    module = hashlib.sha1(module.encode('utf-8')).hexdigest()

    args = semiring._asdict()
    args.update(c_types)
    # set default accumulate operator 
    if accum is None:
        args["accum_binaryop"] = "NoAccumulate"
        args["no_accum"] = "1" 
    else: 
        args["accum_binaryop"] = accum
        args["no_accum"] = "0"
    # set default min identity
    if semiring.add_identity == "MinIdentity":
        args["min_identity"] = "1" 
    else:
        args["min_identity"] = "0"
    return _get_module("accumulate", module, **args)

def _get_module(target, module, **kwargs):
    # first look in dictionary
    try:
        return _gb[module]
    except KeyError:
        # then check directory
        try:
            _gb[module] = importlib.import_module("GraphBLAS.lib." + module)
            return _gb[module]
        # finally build and return module
        except ImportError:
            return _build_module(target, module, **kwargs)

#def _build_module(target, module, dtype, alg, semiring, accum):
def _build_module(target, module, **kwargs):
    # create directory for modules
    if not os.path.exists(_MODDIR):
        os.makedirs(_MODDIR)

    FNULL = open(os.devnull, 'w')
    cmd = [
            "make", 
            target, 
            "MODULE="  + module,
            "PYBIND1=" + _PYBIND[0],
            "PYBIND2=" + _PYBIND[1],
            "PYBIND3=" + _PYBIND[2],
            "PYEXT="   + _PYEXT,
            "DIR="     + _MODDIR + "/"
    ]
    cmd += [arg.upper() + "=" + val for arg, val in kwargs.items()]
    subprocess.call(cmd, cwd=_MODDIR)#, stdout=FNULL)

    # cache module for future access
    _gb[module] = importlib.import_module("lib.%s" % module)
    return _gb[module]
       
def get_type(container):
    # if a is a numpy/scipy array
    try: return container.dtype.type
    # if a is an N-D list/array
    except AttributeError:
        # drill down to data in container
        while type(container) not in _types: 
            container = container[0]
        return type(container)
