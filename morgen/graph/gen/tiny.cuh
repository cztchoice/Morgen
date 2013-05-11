/*
 *   Generate a tiny graph
 *
 *   Copyright (C) 2013 by
 *   Cheng Yichao        onesuperclark@gmail.com
 *
 *   This program is free software; you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation; either version 2 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 */

#include "../csr.cuh"


namespace Morgen {
namespace graph {
namespace gen {

/**
 * Accept a uninitializeed CsrGraph
 * Convert it to a tiny graph
 */
template<typename VertexId, typename SizeT>
int TinyGraph(CsrGraph<VertexId, SizeT> &csr_graph)
{

    SizeT      nodes = 9;
    SizeT      edges = 11;
    SizeT      r[] = {0, 2, 5, 5, 6, 8, 9, 9, 11, 11};
    VertexId   c[] = {1, 3, 0, 2, 4, 4, 5, 7, 8, 6, 8};


    csr_graph.Init(nodes, edges);  

    for (SizeT i = 0; i < nodes + 1; i++) {
        csr_graph.row[i] = r[i];
    }

    for (SizeT i = 0; i < edges; i++) {
        csr_graph.column[i] = c[i];
    }
    
    return 0;
}



} // gen
} // graph
} // Morgen
