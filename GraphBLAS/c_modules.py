import hashlib
import importlib
import inspect
import numpy as np
from collections import OrderedDict
import os
import subprocess
import sys
import zlib

# compiler flags
CXX         = "g++"
LANG	    = "-std=c++14"
OPTS	    = "-O3 -march=native -DNDEBUG -shared -fPIC -fvisibility=hidden"
PICKY	    = "-Wall"
DEBUG 	    = "-g"
FLAGS       = "-DHAVE_CONFIG_H -DGB_USE_SEQUENTIAL"

# project directories
CWD         = inspect.getfile(inspect.currentframe()).rsplit("/", 1)[0]
sys.path.append(CWD)

GB_SOURCE   = "/home/jessecoleman/graphpack/gbtl/src"
MODULES     = os.path.abspath("{cwd}/modules".format(cwd=CWD))
C_CODE      = os.path.abspath("{cwd}/c_code".format(cwd=CWD))
#TODO dynamically configure this with cmake
PROJECT     = "-I{gb_source} -I{c_code}".format(gb_source=GB_SOURCE, c_code=C_CODE)

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


def get_module(target, args, kwargs):

    module = zlib.adler32("t{}a{}k{}".format(target, args, kwargs).encode("utf-8"))
#   module = hashlib.md5(str(args).encode("utf-8")).hexdigest()

    try:
        return importlib.import_module(
                "GraphBLAS.modules.{mod}".format(mod=module)
        )

    except ImportError:
        print("building module {}".format(target))

    if not os.path.exists(MODULES):
        os.makedirs(MODULES)

    cmd = [
            CXX,
            LANG,
            *OPTS.split(),
            *FLAGS.split(),
            *PYBIND.split(),
            #PICKY, DEBUG,
            *PROJECT.split(),
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
            *("-D{arg}".format(
                    arg=str(a).upper()) 
                    for a in args
            ),
            *("-D{key}={arg}".format(
                    key=str(kw).upper(), arg=str(a))
                    for kw, a in kwargs
            )
    ]
    print(cmd)
    subprocess.call(cmd, cwd=C_CODE)

    return importlib.import_module(
            "GraphBLAS.modules.{mod}".format(mod=module)
    )

# decorator class that caches modules in dictionary attached to function
class Cache(dict):

    def __init__(self):
        dict.__init__(self)

    def __getitem__(self, params):

        target, args, kwargs, containers = params

        f_args = dict()  # function parameters
        if isinstance(containers, dict):
            for i, c in containers.items():
                args.append(i + "_" + type(c).__name__)
                kwargs.append((i + "_type", types[c.dtype]))
                f_args[i] = c.container

        module = dict.__getitem__(self, (
            target,
            tuple(sorted(args)), 
            tuple(sorted(kwargs))
        ))

        if len(containers) == 0:
            return module
        else:
            return module, f_args

    def __missing__(self, args):
        module = get_module(*args)
        self[args] = module
        return module
            
module_cache = Cache()
