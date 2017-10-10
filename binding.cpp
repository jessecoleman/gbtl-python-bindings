#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "graphblas/graphblas.hpp"
#include "algorithms/bfs.hpp"

namespace py = pybind11;

template <typename MatrixT, 
          typename IndexT, 
          typename ScalarT>
void define_matrix(py::module &m) {
    py::class_<MatrixT>(m, "Matrix")
        .def(py::init<IndexT, IndexT, ScalarT>())
        .def(py::init<std::vector<std::vector<ScalarT>> const &, ScalarT>())
        .def(py::init<const MatrixT &>())
        .def("get_shape", py::overload_cast<>(&MatrixT::get_shape, py::const_))
        .def("get_shape", py::overload_cast<IndexT &, IndexT &>(&MatrixT::get_shape, py::const_))
        .def("get_zero", &MatrixT::get_zero)
        .def("set_zero", &MatrixT::set_zero)
        .def("get_nnz", &MatrixT::get_nnz)
        .def("__eq__", [](const MatrixT &a, const MatrixT &b) {
                return a == b;
         }, py::is_operator())
        .def("__ne__", [](const MatrixT &a, const MatrixT &b) {
                return a != b;
         }, py::is_operator())
        //.def("__getitem__", [](const py::tuple &args) { 
        //    return MatrixT::get_value_at(x, y); 
        //}, py::is_operator())
        .def("print_info", &MatrixT::print_info);
        //.def
}

//typedef graphblas::Vector<ScalarT, graphblas::
template <typename VectorT, 
          typename OtherVectorT, 
          typename SizeT, 
          typename ScalarT>
void define_vector(py::module &m) {
    py::class_<VectorT>(m, "Vector")
        .def(py::init<OtherVectorT const &>())
        .def(py::init<SizeT const &, ScalarT const &>())
        .def("__eq__", [](const VectorT &a, const VectorT &b) {
                return a == b;
        }, py::is_operator())
        .def("__ne__", [](const VectorT &a, const VectorT &b) {
                return a != b;
        }, py::is_operator());
}

PYBIND11_MODULE(graphBLAS, m) {
    py::class_<graphblas::DirectedMatrixTag>(m, "DirectedMatrixTag");
    py::class_<graphblas::UndirectedMatrixTag>(m, "UndirectedMatrixTag");
    py::class_<graphblas::DenseMatrixTag>(m, "DenseMatrixTag");
    py::class_<graphblas::SparseMatrixTag>(m, "SparseMatrixTag");

    // matrix of doubles
    typedef double ScalarT;
    typedef graphblas::IndexType IndexT;
    // directed matrix
    typedef graphblas::Matrix<ScalarT, graphblas::DirectedMatrixTag> GBMatrix;
    typedef graphblas::Vector<ScalarT> GBVector;

    define_matrix<GBMatrix, IndexT, ScalarT>(m);
    define_vector<GBVector, std::vector<ScalarT>, IndexT, ScalarT>(m);

    typedef graphblas::IndexArrayType::iterator Iterator;
    typedef graphblas::math::Assign<ScalarT> AccumT;
    m.def("build_matrix", py::overload_cast<GBMatrix &, Iterator, Iterator, Iterator, IndexT, AccumT>
            (&graphblas::buildmatrix<GBMatrix &, Iterator, Iterator, Iterator, AccumT>));
    m.def("print_matrix", &graphblas::print_matrix<GBMatrix>);
    m.def("identity", &graphblas::identity<GBMatrix>);
    m.def("bfs", &algorithms::bfs<GBMatrix, GBMatrix, GBMatrix>);

}
