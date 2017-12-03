LANG		= -std=c++14
DEBUG 	    = -g
PICKY		= -Wall 
OPTS		= -O3 -march=native -DNDEBUG -shared -fPIC -fvisibility=hidden
GB_SOURCE 	= /home/jessecoleman/graphpack/gbtl/src
#GB_SOURCE2 	= /home/jessecoleman/graphpack/gbtl/src/src


CXX			= g++
CXXFLAGS	= $(OPTS) $(LANG) 
#PYBIND		= $(shell python3 -m pybind11 --includes)
#PYEXT		= $(shell python3-config --extension-suffix)

test: binding_test.cpp
	$(CXX) $(CXXFLAGS) -o test.out $(PYBIND) $(DIR)binding_test.cpp

container: binding_containers.cpp
	$(CXX) $(CXXFLAGS) -o $(DIR)$(MODULE)$(PYEXT) -DMODULE=$(MODULE) -DDTYPE=$(ATYPE) -DMASK=$(MASK) -DHAVE_CONFIG_H -DGB_USE_SEQUENTIAL -I$(DIR). -I$(GB_SOURCE) $(PYBIND1) $(PYBIND2) $(PYBIND3) -MT graphblas$(PYEXT) -MD -MP -MF $(DIR).deps/binding.Tpo $(DIR)binding_containers.cpp

algorithm: binding_algorithms.cpp
	$(CXX) $(CXXFLAGS) -o $(DIR)$(MODULE)$(PYEXT) -DMODULE=$(MODULE) -DDTYPE=$(DTYPE) -D$(ALG) -DHAVE_CONFIG_H -DGB_USE_SEQUENTIAL -I$(DIR). -I$(GB_SOURCE) $(PYBIND1) $(PYBIND2) $(PYBIND3) -MT graphblas$(PYEXT) -MD -MP -MF $(DIR).deps/binding.Tpo $(DIR)binding_algorithms.cpp

apply: binding_apply.cpp
	$(CXX) $(CXXFLAGS) -o $(DIR)$(MODULE)$(PYEXT) -DMODULE=$(MODULE) -DATYPE=$(ATYPE) -DCTYPE=$(CTYPE) -DAPPLY_OP=$(APPLY_OP) -DBOUND_SECOND=$(BOUND_SECOND) -DBOUND_CONST=$(BOUND_CONST) -DACCUM_BINARYOP=$(ACCUM_BINARYOP) -DNO_ACCUM=$(NO_ACCUM) -DHAVE_CONFIG_H -DGB_USE_SEQUENTIAL -I$(DIR). -I$(GB_SOURCE) $(PYBIND1) $(PYBIND2) $(PYBIND3) -MT graphblas$(PYEXT) -MD -MP -MF $(DIR).deps/binding.Tpo $(DIR)binding_apply.cpp

accumulate: binding_accumulators.cpp
	$(CXX) $(CXXFLAGS) -o $(DIR)$(MODULE)$(PYEXT) -DMODULE=$(MODULE) -DATYPE=$(ATYPE) -DBTYPE=$(BTYPE) -DCTYPE=$(CTYPE) -DADD_BINARYOP=$(ADD_BINARYOP) -DADD_IDENTITY=$(ADD_IDENTITY) -DMIN_IDENTITY=$(MIN_IDENTITY) -DMULT_BINARYOP=$(MULT_BINARYOP) -DACCUM_BINARYOP=$(ACCUM_BINARYOP) -DNO_ACCUM=$(NO_ACCUM) -DHAVE_CONFIG_H -DGB_USE_SEQUENTIAL -I$(DIR). -I$(GB_SOURCE) $(PYBIND1) $(PYBIND2) $(PYBIND3) -MT graphblas$(PYEXT) -MD -MP -MF $(DIR).deps/binding.Tpo $(DIR)binding_accumulators.cpp

utilities: binding_utilities.cpp
	$(CXX) $(CXXFLAGS) -o $(DIR)$(MODULE)$(PYEXT) -DMODULE=$(MODULE) -DHAVE_CONFIG_H -DGB_USE_SEQUENTIAL -I$(DIR). -I$(GB_SOURCE) $(PYBIND1) $(PYBIND2) $(PYBIND3) -MT graphblas$(PYEXT) -MD -MP -MF $(DIR).deps/binding.Tpo $(DIR)binding_utilities.cpp
