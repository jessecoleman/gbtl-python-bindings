#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/numpy.h>

#include "graphblas/graphblas.hpp"

namespace py = pybind11;

// matrix index type
typedef unsigned int IndexT;
typedef GraphBLAS::IndexArrayType IndexArrayT;

// matrix dtype
typedef ATYPE AScalarT;
typedef BTYPE BScalarT;
typedef CTYPE CScalarT;

// matrix storage type
typedef GraphBLAS::Matrix<AScalarT> AMatrixT;
typedef GraphBLAS::Matrix<BScalarT> BMatrixT;
typedef GraphBLAS::Matrix<CScalarT> CMatrixT;
// vector storage type
typedef GraphBLAS::Vector<AScalarT> AVectorT;
typedef GraphBLAS::Vector<BScalarT> BVectorT;
typedef GraphBLAS::Vector<CScalarT> CVectorT;

# ifdef BFS
#include <algorithms/bfs.hpp>
template <typename MatrixT, 
          typename WavefrontMatrixT,
          typename LevelListMatrixT>
LevelListMatrixT bfs_level(MatrixT const &graph, WavefrontMatrixT wavefront) {
    using namespace algorithms;
    LevelListMatrixT parent_list(1, graph.nrows());
    bfs_level(graph, wavefront, parent_list);
    return parent_list;
    //return extract_vector(parent_list);
}

template <typename MatrixT, 
          typename WavefrontMatrixT, 
          typename WavefrontVectorT, 
          typename ParentListVectorT, 
          typename LevelListVectorT>
void define_bfs_algorithms(py::module &m) {
    m.def("bfs_level", &bfs_level<MatrixT, WavefrontMatrixT, MatrixT>);
    //m.def("bfs_level", &algorithms::bfs_level<MatrixT, WavefrontMatrixT, LevelListVectorT>);
    //m.def("bfs_level_masked", &algorithms::bfs_level_masked<MatrixT, WavefrontVectorT, LevelListMatrixT>);
}
#endif

#ifdef MAXFLOW
#include <algorithms/maxflow.hpp>
template <typename MatrixT, typename VectorT>
void define_maxflow_algorithms(py::module &m) {
    using namespace algorithms;
    m.def("push", &push<MatrixT, VectorT>);
    m.def("relabel", &relabel<MatrixT, VectorT>);
    m.def("discharge", &discharge<MatrixT, VectorT>);
    m.def("maxflow", &maxflow<MatrixT>);
}
#endif

#ifdef METRICS
#include <algorithms/metrics.hpp>
template <typename MatrixT, typename VectorT>
void define_metrics_algorithms(py::module &m) {
    m.def("vertex_in_degree", &algorithms::vertex_in_degree<MatrixT>);
    m.def("vertex_out_degree", &algorithms::vertex_out_degree<MatrixT>);
    m.def("vertex_degree", &algorithms::vertex_degree<MatrixT>);
    m.def("graph_distance", &algorithms::graph_distance<MatrixT, VectorT>);
    m.def("graph_distance_matrix", &algorithms::graph_distance_matrix<MatrixT>);
    m.def("vertex_eccentricity", &algorithms::graph_distance_matrix<MatrixT>);
    m.def("graph_radius", &algorithms::graph_radius<MatrixT>);
}
#endif

#ifdef MIS
#include <algorithms/mis.hpp>
template <typename T, typename MatrixT>
void define_mis_algorithms(py::module &m) {
    m.def("get_vertex_IDs", &algorithms::get_vertex_IDs<MatrixT>);
    m.def("mis", &algorithms::mis<MatrixT>);
}
#endif
 
#ifdef MST
#include <algorithms/mst.hpp>
template <typename T, typename MatrixT>
void define_mst_algorithms(py::module &m) {
    //py::class_<LessThan<T>>(m, "LessThan");

    using namespace algorithms;
    m.def("mst", &mst<MatrixT>);
}
#endif

#ifdef PAGERANK
#include <algorithms/page_rank.hpp>
template <typename MatrixT, typename PRMatrixT>
void define_pagerank_algorithms(py::module &m) {
    using namespace algorithms;
    m.def("page_rank", &page_rank<MatrixT, PRMatrixT>);
}
#endif

#ifdef SSSP
#include <algorithms/sssp.hpp>
template <typename MatrixT,
          typename VectorT>
void define_sssp_algorithms(py::module &m) {
    using namespace algorithms;
    m.def("sssp", &sssp<MatrixT, VectorT>);
}
#endif

#ifdef TRICOUNT
#include <algorithms/triangle_count.hpp>
template <typename MatrixT>
void define_tri_count_algorithms(py::module &m) {
    using namespace algorithms;
    m.def("triangle_count", &triangle_count<MatrixT>);
    m.def("triangle_count_newGBTL", &triangle_count_newGBTL<MatrixT, MatrixT>);
}
#endif

PYBIND11_MODULE(MODULE, m) {
#ifdef BFS
    define_bfs_algorithms<GBMatrix, GBMatrix, GBVector, GBVector, GBVector>(m);
#endif
#ifdef MAXFLOW
    define_maxflow_algorithms<GBMatrix, GBVector>(m);
#endif
#ifdef METRICS
    define_metrics_algorithms<GBMatrix, GBVector>(m);
#endif
#ifdef MIS
    define_mis_algorithms<ScalarT, GBMatrix>(m);
#endif
#ifdef MST
    define_mst_algorithms<ScalarT, GBMatrix>(m);
#endif
#ifdef SSSP
    define_sssp_algorithms<GBMatrix, GBVector>(m);
#endif
#ifdef TRICOUNT
    define_tri_count_algorithms<GBMatrix>(m);
#endif
}
