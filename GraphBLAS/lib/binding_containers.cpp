#include <string>
#include <sstream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/numpy.h>

#include <graphblas/graphblas.hpp>
#include <graphblas/ComplementView.hpp>

namespace py = pybind11;

// matrix index type
typedef unsigned int IndexT;
typedef GraphBLAS::IndexArrayType IndexArrayT;

// matrix dtype
typedef DTYPE ScalarT;

// matrix storage type
typedef GraphBLAS::Matrix<ScalarT> MatrixT;
typedef GraphBLAS::Vector<ScalarT> VectorT;

typedef GraphBLAS::MatrixComplementView<MatrixT> MatrixCompT;
typedef GraphBLAS::VectorComplementView<VectorT> VectorCompT;

typedef GraphBLAS::TransposeView<MatrixT> MatrixTransposeT;


MatrixT init_sparse_matrix(
        IndexT &rows, 
        IndexT &cols,
        IndexArrayT &i, 
        IndexArrayT &j, 
        std::vector<ScalarT> const &v); 

VectorT init_sparse_vector(
        IndexT &size,
        IndexArrayT &i, 
        std::vector<ScalarT> const &vals);

template <typename ContainerT>
std::string print(ContainerT const &c);

MatrixTransposeT transpose(MatrixT const &m);

MatrixCompT matrix_complement(MatrixT const &m);

VectorCompT vector_complement(VectorT const &v);

std::vector<ScalarT> extract_vector(VectorT const v);

void define_matrix(py::module &m) 
{
    py::class_<MatrixT>(m, "Matrix")
        .def("nvals", &MatrixT::nvals)
        .def("nrows", &MatrixT::nrows)
        .def("ncols", &MatrixT::nrows)
        .def("hasElement", &MatrixT::hasElement)
        .def("extractElement", &MatrixT::extractElement)
        .def("setElement", &MatrixT::setElement)
        .def("T", &transpose)
        .def("__invert__", &matrix_complement, py::is_operator())
        .def("__str__", &print<MatrixT>, py::is_operator());

    py::class_<MatrixTransposeT>(m, "MatrixTransposeView")
        .def("__str__", print<MatrixTransposeT>);

    py::class_<MatrixCompT>(m, "MatrixComplementView")
        .def("__str__", &print<MatrixCompT>);

    m.def("init_sparse_matrix", &init_sparse_matrix);
}

template <typename OtherVectorT>
void define_vector(py::module &m) 
{
    py::class_<VectorT>(m, "Vector")
        .def(py::init<OtherVectorT const &>())
        .def(py::init<std::vector<ScalarT> const &>())
        .def("nvals", &VectorT::nvals)
        .def("size", &VectorT::size)
        .def("hasElement", &VectorT::hasElement)
        .def("extractElement", &VectorT::extractElement)
        .def("setElement", &VectorT::setElement)
        .def("__invert__", &vector_complement, py::is_operator())
        .def("__str__", &print<VectorT>, py::is_operator())
        .def("__eq__", [](const VectorT &a, const VectorT &b) {
                return a == b;
        }, py::is_operator())
        .def("__ne__", [](const VectorT &a, const VectorT &b) {
                return a != b;
        }, py::is_operator());

    py::class_<VectorCompT>(m, "VectorComplementView")
        .def("__str__", &print<MatrixCompT>);

    m.def("init_sparse_vector", &init_sparse_vector);
}

// initialize matrix and return it
MatrixT init_sparse_matrix(
        IndexT &rows, 
        IndexT &cols,
        IndexArrayT &i, 
        IndexArrayT &j, 
        std::vector<ScalarT> const &vals)
{
    MatrixT m(rows, cols);
    m.build(i, j, vals);
    return m;
}

// initialize vector and return it
VectorT init_sparse_vector(
        IndexT &size,
        IndexArrayT &i, 
        std::vector<ScalarT> const &vals) 
{
    VectorT v(size);
    v.build(i, vals);
    return v;
}

MatrixCompT matrix_complement(MatrixT const &m) 
{ return GraphBLAS::complement(m); }

VectorCompT vector_complement(VectorT const &v) 
{ return GraphBLAS::complement(v); }


template <typename ContainerT>
std::string print(ContainerT const &c)
{
    std::ostringstream stream;
    c.printInfo(stream); 
    return stream.str();
}

MatrixTransposeT transpose(MatrixT const &m)
{ return GraphBLAS::transpose(m); }

std::vector<ScalarT> extract_vector(VectorT v) {
    std::vector<ScalarT> result(v.size(),0);
    for (auto i = 0; i < v.size(); ++i) {
        if (v.hasElement(i)) { 
            result[i] = v.extractElement(i); 
        }
    }
    return result;
}

PYBIND11_MODULE(MODULE, m) {
    define_matrix(m);
    define_vector<VectorT>(m);

#if MASK == 1
    py::class_<GraphBLAS::NoMask>(m, "NoMask")
        .def(py::init<>());
#endif
}
