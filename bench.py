import sys
import subprocess
import numpy as np
from GraphBLAS import *
from GraphBLAS.operators import Replace
from GraphBLAS import c_functions
import math
import networkx as ntx
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import time
import random
import bfs
import page_rank
import sssp
import triangle_count


def bench_build():

    s, py1, py2, py3, cpp1, cpp2, cpp3 = [], [], [], [], [], [], []

    for size in [2**i for i in range(2, 14)]:

        s.append(size)

        begin = time.time()

        p1, p2, p3, c1, c2, c3 = [], [], [], [], [], []

        count = 0

        while time.time() - begin < 60 * 5 and count < 20:
            count += 1

            g = ntx.gnp_random_graph(size, size**(-1/2))
            i, j = zip(*g.edges())
            i, j = list(i), list(j)
            i_temp = i[:]
            i.extend(j)
            j.extend(i_temp)

            with open("graph.txt", 'w') as graph:
                for idx, jdx in zip(i, j):
                    graph.write("{}\t{}\n".format(idx, jdx))

            # benchmark
            s = time.time()

            t1, t2, t3 = c_functions.bench("build")
            c1.append(t1/1000)
            c2.append(t2/1000)
            c3.append(t3/1000)
            print("native:", t1/1000, t2/1000, t3/1000)

            t1, t2, t3 = build_matrix(size)
            p1.append(t1)
            p2.append(t2)
            p3.append(t3)
            print("python:", t1, t2, t3)

            end = time.time() - s
            print(end)
            exit()

        cpp1.append(np.mean(c1))
        cpp2.append(np.mean(c2))
        cpp3.append(np.mean(c3))
        py1.append(np.mean(p1))
        py2.append(np.mean(p2))
        py3.append(np.mean(p3))

    fig, ax = plt.subplots(1, 1)
    width = 0.35

    xpos = np.arange(len(s))

    print(cpp1, cpp2, cpp3)

    cpp_bars = np.array([0] * len(range(6, 14)), dtype=float)
    py_bars = np.array([0] * len(range(6, 14)), dtype=float)

    cpp_bars += np.array([bar.get_height() for bar in ax.bar(xpos, cpp1, width=0.35).get_children()])
    cpp_bars += np.array([bar.get_height() for bar in ax.bar(xpos, cpp2, bottom=cpp1, width=0.35).get_children()])
    bars = ax.bar(xpos, cpp3, bottom=[one + two for one, two in zip(cpp1, cpp2)], width=0.35)
    cpp_bars += np.array([bar.get_height() for bar in bars.get_children()])

    for height, r1 in zip(cpp_bars, bars):
        ax.text(r1.get_x() + r1.get_width()/2., 
                height,                 
                "C++",
                ha='center', 
                va='bottom', size=8)

    py_bars += np.array([bar.get_height() for bar in ax.bar(xpos + width, py1, width=width, label="read", color="C0").get_children()])
    py_bars += np.array([bar.get_height() for bar in ax.bar(xpos + width, py2, bottom=py1, width=width, label="construct", color="C1").get_children()])
    bars = ax.bar(xpos + width, py3, bottom=[one + two for one, two in zip(py1, py2)], width=width, label="extract", color="C2")
    py_bars += np.array([bar.get_height() for bar in bars.get_children()])

    for height, r1 in zip(py_bars, bars):         
        print(height)
        ax.text(r1.get_x() + r1.get_width()/2., 
                height,                 
                "Py",
                ha='center', 
                va='bottom', size=8)


    ax.set_xticks(xpos + width/2) #, [str(size) for size in s])
    ax.set_xticklabels([r"$2^{" + str(size) + "}$" for size in range(6, 16)])
    #ax.set_xscale('log', basex=2)
#    ax.bar(size, py2, width=0.35, bottom=py1, label="Python construction")
#    ax.bar(size, py3, width=0.35, bottom=py2, label="Python extraction")
    h, l = ax.get_legend_handles_labels()
    ax.legend(h,l)
    ax.set_xlabel('Matrix size (log scale)')
    ax.set_ylabel('Time in seconds (log scale)')
    ax.set_title('Moving data around in PyGB')


    pp = PdfPages('build_bench.pdf')
    pp.savefig(fig)
    plt.savefig("{}_bench.png".format("build"))

def build_matrix(l):
    start = time.time()
    idx, jdx = [], []
    with open("graph.txt", 'r') as graph:
        for line in graph:
            i, j = line.strip().split("\t")
            idx.append(int(i))
            jdx.append(int(j))

    v = [1] * len(idx)

    t1 = time.time() - start
    graph = Matrix((v, (idx, jdx)), shape=(l, l))

    t2 = time.time() - (t1 + start)
    t = graph.container.extractTuples()
    t3 = time.time() - (t1 + t2 + start)

    return t1, t2, t3

def time_function(function, *args):
    start = time.time()
    function(*args)
    end = time.time()
    print("Python:", end - start)
    return end - start
 
import re

def cpp_build_matrix(matrix):
    with open("graph1.txt", "w") as m:
        m.write("{}\t{}\t{}\n".format(*matrix.shape, re.search(r"'([^']*)'", str(matrix.dtype)).group(1)))
        for i, j, v in matrix:
            m.write("{}\t{}\t{}\n".format(i, j, v))

def cpp_bfs(root):
    result = subprocess.check_output(["./bench_all", "matrix.txt", str(root)])
    return int(str(result).split()[1][:-1]) / 1000

def bench_bfs():

    x, y1, y1err, y2, y2err, y3, y3err = [], [], [], [], [], [], []

    for i in [2**i for i in range(6, 14)]:

        g = ntx.gnp_random_graph(i, i**(-1/2))
        i, j = zip(*g.edges())
        i, j = list(i), list(j)
        i_temp = i[:]
        i.extend(j)
        j.extend(i_temp)
        v = [1] * len(i)

        m = Matrix((v, (i,j)))
        cpp_build_matrix(m)

        print(m.shape)

        python_time, cpp_time, cpp_native_time = [], [], []

        root = random.randint(0, m.shape[0] - 1)
        print("root", root)
        
        begin = time.time()
        count = 0
        while time.time() - begin < 60 * 8 and count < 50:
            count += 1

            wavefront = Vector(([1], [root]), shape=(m.shape[0],))
            parentlist = Vector(shape=(m.shape[0],), dtype=int)

            t = time_function(bfs.bfs_level_masked_v2, m, wavefront, parentlist)
            python_time.append(t)

            temp = parentlist

            wavefront = Vector(([1], [root]), shape=(m.shape[0],))
            parentlist = Vector(shape=(m.shape[0],), dtype=int)

            t = time_function(algorithms.bfs_level_masked_v2, m, wavefront, parentlist)
            cpp_time.append(t)

            wavefront = Vector(([1], [root]), shape=(m.shape[0],))
            parentlist = Vector(shape=(m.shape[0],), dtype=int)

            t = c_functions.bench(
                    "bfs", 
                    graph=m, 
                    wavefront=wavefront, 
                    levels=parentlist
            ) / 1000
            print("native ", t)
            cpp_native_time.append(t)

            # error check
            for i, j in zip(iter(temp), iter(parentlist)):
                if i != j:
                    print('ERROR')

        print("Python Total:", np.mean(python_time))
        print("CPP Total:", np.mean(cpp_time))
        print("Native Total:", np.mean(cpp_native_time))

        x.append(m.shape[0])
        y1.append(np.mean(python_time))
        y1err.append(np.std(python_time))
        y2.append(np.mean(cpp_time))
        y2err.append(np.std(cpp_time))
        y3.append(np.mean(cpp_native_time))
        y3err.append(np.std(cpp_native_time))


    print(x, y1, y2)
    
    x, y1, y1err, y2, y2err, y3, y3err = zip(*sorted(zip(x, y1, y1err, y2, y2err, y3, y3err)))
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.errorbar(x, y1, yerr=y1err, capsize=2, label="Python loops")
    ax.errorbar(x, y2, yerr=y3err, capsize=2, label="C++ loops")
    ax.errorbar(x, y3, yerr=y3err, capsize=2, label="C++ native")
    ax.set_xscale('log', basex=2)
    ax.set_yscale('log')
    ax.set_xlabel('Matrix size (log scale)')
    ax.set_ylabel('Time in seconds (log scale)')
    ax.set_title('Breadth-first search algorithm')

    h, l = ax.get_legend_handles_labels()
    ax.legend(h,l)
    
    pp = PdfPages('bfs_rank_bench.pdf')
    pp.savefig(fig)
    plt.savefig("{}_bench.png".format("bfs"))
 
def bench_page_rank():

    x, y1, y1err, y2, y2err, y3, y3err = [], [], [], [], [], [], []

    for i in [2**i for i in range(6, 14)]:

        g = ntx.gnp_random_graph(i, i**(-1/2))
        i, j = zip(*g.edges())
        i, j = list(i), list(j)
        i_temp = i[:]
        i.extend(j)
        j.extend(i_temp)
        v = [1] * len(i)

        m = Matrix((v, (i,j)))
        cpp_build_matrix(m)

        print(m.shape)

        python_time, cpp_time, cpp_native_time = [], [], []

        root = random.randint(0, m.shape[0] - 1)
        print("root", root)
        
        begin = time.time()
        count = 0
        while time.time() - begin < 60 * 8 and count < 50:
            count += 1

            rank = Vector(shape=(m.shape[0],), dtype=float)

            t = time_function(page_rank.page_rank, m, rank)
            python_time.append(t)

            temp = rank

            rank = Vector(shape=(m.shape[0],), dtype=float)

            t = time_function(algorithms.page_rank, m, rank)
            cpp_time.append(t)

            rank = Vector(shape=(m.shape[0],), dtype=float)

            t = c_functions.bench(
                    "page_rank", 
                    graph=m, 
                    page_rank=rank, 
            ) / 1000
            print("native ", t)
            cpp_native_time.append(t)

            # error check
            for i, j in zip(iter(temp), iter(rank)):
                if i != j:
                    print('ERROR')

        print("Python Total:", np.mean(python_time))
        print("CPP Total:", np.mean(cpp_time))
        print("Native Total:", np.mean(cpp_native_time))

        x.append(m.shape[0])
        y1.append(np.mean(python_time))
        y1err.append(np.std(python_time))
        y2.append(np.mean(cpp_time))
        y2err.append(np.std(cpp_time))
        y3.append(np.mean(cpp_native_time))
        y3err.append(np.std(cpp_native_time))


    x, y1, y1err, y2, y2err, y3, y3err = zip(*sorted(zip(x, y1, y1err, y2, y2err, y3, y3err)))
   
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.errorbar(x, y1, yerr=y1err, capsize=2, label="Python loops")
    ax.errorbar(x, y2, yerr=y3err, capsize=2, label="C++ loops")
    ax.errorbar(x, y3, yerr=y3err, capsize=2, label="C++ native")
    ax.set_xscale('log', basex=2)
    ax.set_yscale('log')
    ax.set_xlabel('Matrix size (log scale)')
    ax.set_ylabel('Time in seconds (log scale)')
    ax.set_title('PageRank algorithm')

    h, l = ax.get_legend_handles_labels()
    ax.legend(h,l)
    
    pp = PdfPages('page_rank_bench.pdf')
    pp.savefig(fig)
    plt.savefig("{}_bench.png".format("page_rank"))

def bench_sssp():

    x, y1, y1err, y2, y2err, y3, y3err = [], [], [], [], [], [], []

    for i in [2**i for i in range(6, 12)]:

        g = ntx.gnp_random_graph(i, i**(-1/2))
        i, j = zip(*g.edges())
        i, j = list(i), list(j)
        i_temp = i[:]
        i.extend(j)
        j.extend(i_temp)
        v = [1] * len(i)

        m = Matrix((v, (i,j)))
        cpp_build_matrix(m)

        print(m.shape)

        python_time, cpp_time, cpp_native_time = [], [], []

        root = random.randint(0, m.shape[0] - 1)
        print("root", root)
        
        begin = time.time()
        count = 0
        while time.time() - begin < 60 * 8 and count < 50:
            count += 1

            path = Vector(shape=(m.shape[0],), dtype=int)
            path[0] = 0

            t = time_function(sssp.sssp, m, path)
            python_time.append(t)

            temp = path

            path = Vector(shape=(m.shape[0],), dtype=float)
            path[0] = 0

            t = time_function(algorithms.sssp, m, path)
            cpp_time.append(t)

            path = Vector(shape=(m.shape[0],), dtype=float)
            path[0] = 0

            t = c_functions.bench(
                    "sssp", 
                    graph=m, 
                    path=path, 
            ) / 1000
            print("native ", t)
            cpp_native_time.append(t)


            # error check
            for i, j in zip(iter(temp), iter(path)):
                if i != j:
                    print('ERROR')

        print("Python Total:", np.mean(python_time))
        print("CPP Total:", np.mean(cpp_time))
        print("Native Total:", np.mean(cpp_native_time))

        x.append(m.shape[0])
        y1.append(np.mean(python_time))
        y1err.append(np.std(python_time))
        y2.append(np.mean(cpp_time))
        y2err.append(np.std(cpp_time))
        y3.append(np.mean(cpp_native_time))
        y3err.append(np.std(cpp_native_time))

    x, y1, y1err, y2, y2err, y3, y3err = zip(*sorted(zip(x, y1, y1err, y2, y2err, y3, y3err)))
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.errorbar(x, y1, yerr=y1err, capsize=2, label="Python loops")
    ax.errorbar(x, y2, yerr=y3err, capsize=2, label="C++ loops")
    ax.errorbar(x, y3, yerr=y3err, capsize=2, label="C++ native")
    ax.set_xscale('log', basex=2)
    ax.set_yscale('log')
    ax.set_xlabel('Matrix size (log scale)')
    ax.set_ylabel('Time in seconds (log scale)')
    ax.set_title('SSSP algorithm')

    h, l = ax.get_legend_handles_labels()
    ax.legend(h,l)
    
    pp = PdfPages('sssp_bench.pdf')
    pp.savefig(fig)
    plt.savefig("{}_bench.png".format("sssp"))

def bench_tri():

    x, y1, y1err, y2, y2err, y3, y3err = [], [], [], [], [], [], []

    for i in [2**i for i in range(6, 14)]:

        g = ntx.gnp_random_graph(i, i**(-1/2))
        i, j = zip(*g.edges())
        i, j = list(i), list(j)
        i_temp = i[:]
        i.extend(j)
        j.extend(i_temp)
        v = [1] * len(i)

        m = Matrix((v, (i,j)))
        cpp_build_matrix(m)

        print(m.shape)

        python_time, cpp_time, cpp_native_time = [], [], []

        root = random.randint(0, m.shape[0] - 1)
        print("root", root)
        
        begin = time.time()
        count = 0
        while time.time() - begin < 60 * 8 and count < 50:
            count += 1

            t = time_function(triangle_count.triangle_count_masked, m)
            python_time.append(t)

            t = time_function(algorithms.triangle_count_masked, m)
            cpp_time.append(t)

            t = c_functions.bench(
                    "triangle_count_masked", 
                    graph=m, 
            ) / 1000
            print("native ", t)
            cpp_native_time.append(t)

        print("Python Total:", np.mean(python_time))
        print("CPP Total:", np.mean(cpp_time))
        print("Native Total:", np.mean(cpp_native_time))

        x.append(m.shape[0])
        y1.append(np.mean(python_time))
        y1err.append(np.std(python_time))
        y2.append(np.mean(cpp_time))
        y2err.append(np.std(cpp_time))
        y3.append(np.mean(cpp_native_time))
        y3err.append(np.std(cpp_native_time))

    x, y1, y1err, y2, y2err, y3, y3err = zip(*sorted(zip(x, y1, y1err, y2, y2err, y3, y3err)))
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.errorbar(x, y1, yerr=y1err, capsize=2, label="Python loops")
    ax.errorbar(x, y2, yerr=y3err, capsize=2, label="C++ loops")
    ax.errorbar(x, y3, yerr=y3err, capsize=2, label="C++ native")
    ax.set_xscale('log', basex=2)
    ax.set_yscale('log')
    ax.set_xlabel('Matrix size (log scale)')
    ax.set_ylabel('Time in seconds (log scale)')
    ax.set_title('Triangle count algorithm')

    h, l = ax.get_legend_handles_labels()
    ax.legend(h,l)
    
    pp = PdfPages('tricount_bench.pdf')
    pp.savefig(fig)
    plt.savefig("{}_bench.png".format("tricount"))


def bench_read_fromfile():

    i = 10

    g = ntx.gnp_random_graph(i, i**(-1/2))
    i, j = zip(*g.edges())
    i, j = list(i), list(j)
    i_temp = i[:]
    i.extend(j)
    j.extend(i_temp)
    v = [1] * len(i)

    m = Matrix((v, (i,j)))
    print("M1:", m)
    cpp_build_matrix(m)

    start = time.time()
    idx, jdx, vals = [], [], []
    with open("graph1.txt", 'r') as graph:

        rows, cols, dtype = graph.readline().strip().split()
        rows, cols = int(rows), int(cols)
        dtype = eval(dtype)

        print(rows, cols, dtype)

        for line in graph:
            i, j, v = line.strip().split()
            idx.append(int(i))
            jdx.append(int(j))
            vals.append(float(v))

    print(len(i), len(j), len(v))

    graph = Matrix((vals, (idx, jdx)), shape=(rows, cols))
    print("M2:", graph)
    py_time = time.time() - start
    #print(graph)

    start = time.time()
    m = Matrix("graph1.txt")
    print("M3:", m)
    cpp_time = time.time() - start
    #print(m)


if __name__ == '__main__':
    if sys.argv[1] == 'density':
        bench_density()

    elif sys.argv[1] == 'bfs':
        bench_bfs()

    elif sys.argv[1] == 'page':
        bench_page_rank()

    elif sys.argv[1] == 'sssp':
        bench_sssp()

    elif sys.argv[1] == 'tri':
        bench_tri()

    elif sys.argv[1] == 'build':
        bench_build()

    elif sys.argv[1] == 'read':
        bench_read_fromfile()
