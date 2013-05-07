/*
 *   A test for the performance of atomic operation on GPU
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


#include <argvparser/argvparser.h>
#include <cstdlib>
#include <iostream>


using namespace CommandLineProcessing;
using namespace std;


/********************************************************************************
 * Globals
 ********************************************************************************/

int      g_num_stride_log       = 1;     
int      g_num_elements         = 1024;
int      g_num_iterations       = 1;
bool     g_local                = false;



/**********************************************************************************
 * Init parameters
 **********************************************************************************/

void Init(int argc, char** argv) {

    ArgvParser cmd;

    // help option
    cmd.setHelpOption("help", "h", "help menu");

    // options with value
    cmd.defineOption("stride", "stride of writing atomically",
                     ArgvParser::OptionRequiresValue); 
    cmd.defineOption("elements", "how many numbers to be performed on",
                     ArgvParser::OptionRequiresValue);
    cmd.defineOption("iterations", "how many iterations to be performed",
                     ArgvParser::OptionRequiresValue);
    
    // without value
    cmd.defineOption("local", "use local kernel(or global kernel)");

    ArgvParser::ParserResults result = cmd.parse(argc, argv);


    if (result != ArgvParser::NoParserError) {
        cout << cmd.parseErrorDescription(result)
             << "  (type -h for help) \n";
        exit(1);
    }

    
    if (cmd.foundOption("stride")) {
        const char* s = cmd.optionValue("stride").c_str();
        g_num_strides_log = atoi(s);
    }


    if (cmd.foundOption("elements")) {
        const char* s = cmd.optionValue("elements").c_str();
        g_num_elements = atoi(s);
    }

    if (cmd.foundOption("iterations")) {
        const char* s = cmd.optionValue("blocks").c_str();
        g_num_iterations = atoi(s);
    }

    
    if (cmd.foundOption("local")) {
        g_local = true;
    }

}


/************************************************************************************
 * Kernels
 ************************************************************************************/

__global__ void GlobalAtomicKernel(
    int *d_counter,
    int *d_in,
    int *d_out,
    int stride,
    int tiles_per_block,
    int extra_tiles) 
{

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int temp;
    
    /*
     * extra_tiles == 3 means only 3 blocks of threads need to
     * process the extra tiles. So increase the tiles
     */
    if (blockIdx.x < extra_tiles) {
        tiles_per_block++;
    } 


    for (int i = 0; i < tiles_per_block; i++) {
        temp = d_in[tid];
        
        if (tid % stride == 0) {
            // atomic opertion with stride
            atomicAdd(d_counter, 1);
        }
        
        d_out[tid] = temp;

        tid += blockDim.x;
    }
}


__global__ void LocalAtomicKernel(
    int *d_counter,
    int *d_in,
    int *d_out,
    int stride,
    int tiles_per_block,
    int extra_tiles) 
{


}



/*************************************************************************************
 * Main
 *************************************************************************************/

int main(int argc, char** argv) 
{

    Init(argc, argv);

    printf("strides(log): %d \n"         
           "elements: %d     \n"
           "local: %b        \n",
           "iterations: %d   \n",
           g_num_strides_log,
           g_num_elements,
           g_local,
           g_num_iterations);


    // host alloc
    int *in = new int[g_num_elements];
    int *out = new int[g_num_elements];
    
    util::RandomizeArray(in, g_num_elements);
    util::PrintArray(in, g_num_elements);


    // device alloc
    int *d_counter = NULL;   // global memory counter
    int *d_in = NULL;
    int *d_out = NULL;

    cudaMalloc((void**) &d_counter, sizeof(int) * 1);
    cudaMalloc((void**) &d_in, sizeof(int) * g_num_elements);
    cudaMalloc((void**) &d_out, sizeof(int) * g_num_elements);

    if (NULL == d_counter || NULL == d_in || NULL == d_out) {
        printf("cudamalloc fail!\n");
        exit(1);
    }


    // copy data_in to device
    cudaMemcpy(d_in, in, sizeof(int) * g_num_elements,
               cudaMemcpyHostToDevice);


    /*
     * NOTE:
     * We will enlist as much blocks as possible to process tiles
     * but no more than MAX_NUM_BLOCKS.
     * The extra tiles will be assigned iteratively in the kernel,
     * which means tiles_per_block > 1
     * In some cases, tiles cannot be divided by num_blocks,
     * so we need handle this situation by extra_tiles
     */

    const int THREADS_PER_BLOCK = 256;
    const int MAX_NUM_BLOCKS = 1024 * 16;    

    int tiles = (g_num_elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    int num_blocks = (tiles > MAX_NUM_BLOCKS) ?
        MAX_NUM_CTAS :
        tiles;
    
    int tiles_per_block = tiles / num_blocks;
    int extra_tiles = tiles - (tiles_per_block * num_blocks);


    // iteration starts
    for (int i = 0; i < g_iterations; i++) {
        
        util::GPUTimer timer;

        timer.Start();

        if(g_local) {
            LocalAtomicKernel <<< num_blocks, THREADS_PER_BLOCK >>>(
                d_in,
                d_out,
                stride,
                tiles_per_block,
                extra_tiles);
        } else {
            GlobalAtomicKernel <<< num_blocks, THREADS_PER_BLOCK >>>(
                d_in,
                d_out,
                stride,
                tiles_per_block,
                extra_tiles);
        }        

        timer.Stop();
        
        float millis = timer.ElapsedTime();
        float atomics = float(g_elements) / g_num_strides;
        unsigned long long bytes = g_num_elements * sizeof(int) * 2;

        printf("%.5f 10^9 atomics/sec \n",
               atomics / millis / 1000.0 / 1000.0);
        printf("%.5f 10^9 bytes/sec   \n",
               float(bytes) / millis / 1000.0 / 1000.0);

    }
    


    // copy data_out to device
    cudaMemcpy(out, d_out, sizeof(int) * g_num_elements,
               cudaMemcpyHostToDevice);

    util::PrintArray(d, g_num_elements);


    // cleaning
    if (d_counter) cudaFree(d_counter);
    if (d_in) cudaFree(d_in);
    if (d_out) cudaFree(d_out);
    
    delete[] in;
    delete[] out;

    return 0;
}
