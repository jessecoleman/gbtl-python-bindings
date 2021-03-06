#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>

#include <graphblas/graphblas.hpp>
//#include <graphblas.hpp>


namespace py = pybind11;
using namespace pybind11::literals;

// matrix index type
typedef unsigned int IndexT;
typedef GraphBLAS::IndexArrayType IndexArrayT;

// BFS types
#if defined(BFS)
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
#elif defined(WAVEFRONT_TYPE) && defined(LEVELS_TYPE)
typedef GraphBLAS::Vector<WAVEFRONT_TYPE> WavefrontMatrixT;
typedef GraphBLAS::Vector<LEVELS_TYPE> LevelsMatrixT;

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
typedef GraphBLAS::Matrix<GRAPH_TYPE> MatrixT;
typedef GraphBLAS::Vector<INDEPENDENT_SET_TYPE> VectorT;
#elif defined(MST)
#include <algorithms/mst.hpp>
#elif defined(PAGE_RANK)
#include <algorithms/page_rank.hpp>
typedef GraphBLAS::Matrix<GRAPH_TYPE> MatrixT;
typedef PAGE_RANK_TYPE RealT;
#elif defined(SSSP)
#include <algorithms/sssp.hpp>
typedef GraphBLAS::Matrix<GRAPH_TYPE> MatrixT;
typedef GraphBLAS::Vector<PATH_TYPE> PathT;

#elif defined(TRIANGLE_COUNT)
#include <algorithms/triangle_count.hpp>
#if defined(GRAPH_TYPE)
typedef GraphBLAS::Matrix<GRAPH_TYPE> MatrixT;
#elif defined(L_TYPE)
typedef GraphBLAS::Matrix<L_TYPE> MatrixT;
#endif

#endif

PYBIND11_MODULE(MODULE, m) {
#if defined(WAVEFRONT_TYPE) && defined(PARENT_LIST_TYPE)
    m.def("bfs", &algorithms::bfs<GraphT, WavefrontVectorT, ParentListVectorT>, "graph"_a, "wavefront"_a, "parent_list"_a);
#elif defined(WAVEFRONTS_TYPE) && defined(PARENT_LIST_TYPE)
    m.def("bfs_batch", &algorithms::bfs_batch<GraphT, WavefrontMatrixT, ParentListMatrixT>, "graph"_a, "wavefronts"_a, "parent_list"_a);
#elif defined(WAVEFRONT_TYPE) && defined(LEVEL_LIST_TYPE)
    m.def("bfs_level", &algorithms::bfs_level<GraphT, WavefrontMatrixT, LevelListMatrixT>, "graph"_a, "wavefront"_a, "level_list"_a);
#elif defined(WAVEFRONT_TYPE) && defined(LEVELS_TYPE)
    m.def("bfs_level_masked_v2", &algorithms::bfs_level_masked_v2<GraphT, WavefrontMatrixT, LevelsMatrixT>, "graph"_a, "wavefront"_a, "levels"_a);
#elif defined(MAXFLOW)
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
    m.def("mis", &algorithms::mis<MatrixT>, "graph"_a, "independent_set"_a, "seed"_a);
#elif defined(MST)
    m.def("mst", &algorithms::mst<MatrixT>, "graph"_a, "mst_parents"_a);
#elif defined(PAGE_RANK)
    m.def("page_rank", &algorithms::page_rank<MatrixT, RealT>, 
            "graph"_a, 
            "page_rank"_a, 
            "damping_factor"_a = 0.85, 
            "threshold"_a = 1.e-5, 
            "max_iters"_a = std::numeric_limits<unsigned int>::max()
    );
#elif defined(SSSP)
    m.def("sssp", &algorithms::sssp<MatrixT, PathT>, "graph"_a, "path"_a);
//    m.def("batch_sssp", &algorithms::sssp<MatrixT, PathMatrixT>, "graph"_a, "paths"_a);
#elif defined(TRIANGLE_COUNT)
    m.def("triangle_count", &algorithms::triangle_count<MatrixT>, "graph"_a);
    m.def("triangle_count_masked", &algorithms::triangle_count_masked<MatrixT>, "graph"_a);
    //m.def("triangle_count_flame1", &algorithms::triangle_count<MatrixT>, "graph"_a);
    //m.def("triangle_count_flame1a", &algorithms::triangle_count<MatrixT>, "graph"_a);
    //m.def("triangle_count_flame2", &algorithms::triangle_count<MatrixT>, "graph"_a);
    //m.def("triangle_count_newGBTL", &algorithms::triangle_count_newGBTL<LMatrixT, MatrixT>, "L"_a, "U"_a);
#endif
}
