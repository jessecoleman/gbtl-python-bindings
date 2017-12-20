#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>

#include <graphblas/graphblas.hpp>

namespace py = pybind11;
using namespace pybind11::literals;

// matrix index type
typedef unsigned int IndexT;
typedef GraphBLAS::IndexArrayType IndexArrayT;

// BFS types
#if defined(BFS_VARIANTS)
#include <algorithms/bfs.hpp>
typedef GraphBLAS::Matrix<GRAPH_TYPE> GraphT;
#endif
#if defined(WAVEFRONT_TYPE) && defined(PARENT_LIST_TYPE)
typedef GraphBLAS::Vector<WAVEFRONT_TYPE> WavefrontVectorT;
typedef GraphBLAS::Vector<PARENT_LIST_TYPE> ParentListVectorT;
#elif defined(WAVEFRONTS_TYPE) && defined(PARENT_LIST_TYPE)
typedef GraphBLAS::Matrix<WAVEFRONTS_TYPE> WavefrontMatrixT;
typedef GraphBLAS::Matrix<PARENT_LIST_TYPE> ParentListMatrixT;
#elif defined(WAVEFRONT_TYPE) && defined(LEVEL_LIST_TYPE)
typedef GraphBLAS::Matrix<WAVEFRONT_TYPE> WavefrontMatrixT;
typedef GraphBLAS::Matrix<LEVEL_LIST_TYPE> LevelListMatrixT;

#elif defined(MAXFLOW)
#include <algorithms/maxflow.hpp>
typedef GraphBLAS::Matrix<GRAPH_TYPE> MatrixT;
#ifdef RESULT_TYPE
typedef GraphBLAS::Vector<RESULT_TYPE> VectorT;
#else
typedef GraphBLAS::Vector<GRAPH_TYPE> VectorT;
#endif

#elif defined(METRICS)
#include <algorithms/metrics.hpp>
typedef GraphBLAS::Matrix<GRAPH_TYPE> MatrixT;
#ifdef RESULT_TYPE
typedef GraphBLAS::Vector<RESULT_TYPE> VectorT;
#else
typedef GraphBLAS::Vector<GRAPH_TYPE> VectorT;
#endif

#elif defined(MIS)
#include <algorithms/mis.hpp>
#elif defined(MST)
#include <algorithms/mst.hpp>
#elif defined(PAGERANK)
#include <algorithms/page_rank.hpp>
#elif defined(SSSP)
#include <algorithms/sssp.hpp>
#elif defined(TRICOUNT)
#include <algorithms/tricount.hpp>
#endif

PYBIND11_MODULE(MODULE, m) {
#if defined(WAVEFRONT_TYPE) && defined(PARENT_LIST_TYPE)
    m.def("bfs", &algorithms::bfs<GraphT, WavefrontVectorT, ParentListVectorT>, "graph"_a, "wavefront"_a, "parent_list"_a);
#elif defined(WAVEFRONTS_TYPE) && defined(PARENT_LIST_TYPE)
    m.def("bfs_batch", &algorithms::bfs_batch<GraphT, WavefrontMatrixT, ParentListMatrixT>, "graph"_a, "wavefronts"_a, "parent_list"_a);
#elif defined(WAVEFRONT_TYPE) && defined(LEVEL_LIST_TYPE)
    m.def("bfs_level", &algorithms::bfs_level<GraphT, WavefrontMatrixT, LevelListMatrixT>, "graph"_a, "wavefront"_a, "level_list"_a);
#elif defined(MAXFLOW)
    m.def("push",       &algorithms::push<MatrixT, VectorT>, "C"_a, "F"_a, "excess"_a, "u"_a, "v"_a);
    m.def("relabel",    &algorithms::relabel<MatrixT, VectorT>, "C"_a, "F"_a, "height"_a, "u"_a);
    m.def("discharge",  &algorithms::discharge<MatrixT, VectorT>, "C"_a, "F"_a, "excess"_a, "height"_a, "seen"_a, "u"_a);
    m.def("maxflow",    &algorithms::maxflow<MatrixT>, "capacity"_a, "source"_a, "sink"_a);
#elif defined(METRICS)
    m.def("vertex_in_degree",       &algorithms::vertex_in_degree<MatrixT>, "graph"_a, "vid"_a);
    m.def("vertex_out_degree",      &algorithms::vertex_out_degree<MatrixT>, "graph"_a, "vid"_a);
    m.def("vertex_degree",          &algorithms::vertex_degree<MatrixT>, "graph"_a, "vid"_a);
    m.def("graph_distance",         &algorithms::graph_distance<MatrixT, VectorT>, "graph"_a, "sid"_a, "result"_a);
    m.def("graph_distance_matrix",  &algorithms::graph_distance_matrix<MatrixT>, "graph"_a, "result"_a);
    m.def("vertex_eccentricity",    &algorithms::graph_distance_matrix<MatrixT>, "graph"_a, "vid"_a);
    m.def("graph_radius",           &algorithms::graph_radius<MatrixT>, "graph"_a);
    m.def("graph_diameter",         &algorithms::graph_diameter<MatrixT>, "graph"_a);
    m.def("closeness_centrality",   &algorithms::closeness_centrality<MatrixT>, "graph"_a, "vid"_a);
#elif defined(MIS)
    m.def("get_vertex_IDs", &algorithms::get_vertex_IDs<MatrixT>, "independent_set"_a);
    m.def("mis", &algorithms::mis<MatrixT>, "graph"_a, "independent_set"_a, "seed"_a=0);
#elif defined(MST)
    m.def("mst", &algorithms::mst<MatrixT>, "graph"_a, "mst_parents"_a);
#elif defined(PAGERANK)
    m.def("page_rank", &page_rank<MatrixT, double>, 
            "graph"_a, 
            "page_rank"_a, 
            "damping_factor"_a = 0.85, 
            "threshold"_a = 1.e-5, 
            "max_iters"_a = std::numeric_limits<unsigned int>::max()
    );
#elif defined(SSSP)
    m.def("sssp", &algorithms::sssp<MatrixT, PathVectorT>, "graph"_a, "path"_a);
    m.def("batch_sssp", &algorithms::sssp<MatrixT, PathMatrixT>, "graph"_a, "paths"_a);
#elif defined(TRICOUNT)
    m.def("triangle_count", &algorithms::triangle_count<MatrixT>, "graph"_a);
    m.def("triangle_count_masked", &algorithms::triangle_count<MatrixT>, "L"_a);
    m.def("triangle_count_flame1", &algorithms::triangle_count<MatrixT>, "graph"_a);
    m.def("triangle_count_flame1a", &algorithms::triangle_count<MatrixT>, "graph"_a);
    m.def("triangle_count_flame2", &algorithms::triangle_count<MatrixT>, "graph"_a);
    m.def("triangle_count_newGBTL", &algorithms::triangle_count_newGBTL<LMatrixT, MatrixT>, "L"_a, "U"_a);
#endif
}
