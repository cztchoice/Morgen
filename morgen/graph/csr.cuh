/*
 *   Graph Representation on GPU
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


#include <morgen/utils/handle_error.cuh>
#include <morgen/utils/print_value.cuh>




namespace Morgen {
namespace graph {


/**************************************************************************
 * CSR - graph data structures
 **************************************************************************/
template<typename VertexId, typename SizeT>
struct CsrGraph
{
    SizeT        node_num;
    SizeT        edge_num;
    SizeT        *row;
    VertexId     *column;
    bool         pinned;   

    /*
     * constructor(default not pinned)     
     */
    CsrGraph(bool pinned = false) :
            node_num(0),
            edge_num(0),
            row(NULL),
            column(NULL),
            pinned(pinned) {}

    
    /** 
     * if pinned is true, allocate on the pinned memory
     * else allocate on the normal memory
     */
    void Init(SizeT nodes, SizeT edges) {

        node_num = nodes;
        edge_num = edges;

        if (pinned) {
            int mapped = cudaHostAllocMapped;        
            if (Morgen::util::HandleError(cudaHostAlloc((void**)&row, sizeof(SizeT) * (node_num + 1), mapped),
                                          "CsrGraph cudaHostAlloc row failed",
                                          __FILE__,
                                          __LINE__)) exit(1);
        
            if (Morgen::util::HandleError(cudaHostAlloc((void**)&column, sizeof(VertexId) * edge_num, mapped),
                                          "CsrGraph cudaHostAlloc column failed",
                                          __FILE__,
                                          __LINE__)) exit(1);  

        } else {
            row      = (SizeT*) malloc(sizeof(SizeT) * (node_num + 1));
            column   = (VertexId*) malloc(sizeof(VertexId) * edge_num); 
        }
    }




    /**
     * Display graph in the console
     */    
    void Display() {
        printf("Displaying Graph:\n");
        
        if (pinned)
            printf("Allocated in pinned memory\n");
        else 
            printf("Allocated in ordinary memory\n");
        

        for (SizeT node = 0; node < node_num; node++) {
            Morgen::util::PrintValue(node);
            printf(" ->");
            for (SizeT edge = row[node]; edge < row[node+1]; edge++) {
                printf("\t");
                Morgen::util::PrintValue(column[edge]);                
            }
            printf("\n");
        }
    }


    /**
     * Display out-degree number of each node in the cosole
     * count the number in log style
     */
    void DisplayOutDegree() {
        printf("Displaying out degree:\n");
        
        int log_counts[32];
        for (int i = 0; i < 32; i++) {
            log_counts[i] = 0;
        }
        
        int max_times = -1;

        for (SizeT node = 0; node < node_num; node++) {

            SizeT degree = row[node + 1] - row[node];
            
            int times = 0;

            while (degree > 0) {
                degree /= 2;  
                times++;                
            }

            if (times > max_times) {
                max_times = times;
            }

            log_counts[times]++;
        }
        
        for (int i = -1; i < max_times + 1; i++) {
            printf("Degree 2^%i: %d (%.2f%%)\n",
                   i,
                   log_counts[i + 1],
                   (float) log_counts[i + 1] * 100.0 / node_num);
        }

        printf("%lld vertices, %lld edges\n",
               (long long) node_num,
               (long long) edge_num);        

    }


    /**
     * explictly free
     */
    void Free() {
        if (pinned) {
            Morgen::util::HandleError(cudaFreeHost(row),
                                      "CsrGraph cudaFreeHost row failed",
                                      __FILE__,
                                      __LINE__);
            Morgen::util::HandleError(cudaFreeHost(column),
                                      "CsrGraph cudaFreeHost column failed",
                                      __FILE__,
                                      __LINE__);                       
        } else {
            free(row);
            free(column);
        }
        row = NULL;
        column = NULL;
        node_num = 0;
        edge_num = 0;
    }


    ~CsrGraph(){
        Free();
    }
    

};


} // namespace graph
} // namespace Morgen
