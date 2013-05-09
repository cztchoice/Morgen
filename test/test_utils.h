/*
 *   Some utilites for testing  
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



#include <stdio.h>
#include <iostream>


namespace Morgen {

namespace util {


/*****************************************************************
 * Print a value(any datatype)
 *****************************************************************/

template<typename T> 
void PrintValue(T val) {
	val.Print();
}


// override it when the data is a basic data type
template<>
void PrintValue<char>(char val) {
	printf("%d", val);
}

template<>
void PrintValue<short>(short val) {
	printf("%d", val);
}

template<>
void PrintValue<int>(int val) {
	printf("%d", val);
}

template<>
void PrintValue<long>(long val) {
	printf("%ld", val);
}

template<>
void PrintValue<long long>(long long val) {
	printf("%lld", val);
}

template<>
void PrintValue<float>(float val) {
	printf("%f", val);
}

template<>
void PrintValue<double>(double val) {
	printf("%f", val);
}

template<>
void PrintValue<unsigned char>(unsigned char val) {
	printf("%u", val);
}

template<>
void PrintValue<unsigned short>(unsigned short val) {
	printf("%u", val);
}

template<>
void PrintValue<unsigned int>(unsigned int val) {
	printf("%u", val);
}

template<>
void PrintValue<unsigned long>(unsigned long val) {
	printf("%lu", val);
}

template<>
void PrintValue<unsigned long long>(unsigned long long val) {
	printf("%llu", val);
}



/*****************************************************************
 * Randomize an array
 *****************************************************************/
template <typename T>
void RandomizeArray(T* a, int len) {
    srand(time(0));
    for (int i = 0; i < len; i++) {
        a[i] = (T) (rand() % 65536);   // rand() returns an integer
    }

}




/******************************************************************
 * Print an array
 ******************************************************************/

template <typename T>
void PrintArray(T* a, int len)
{
    for (int i = 0; i < len; i++) {
        PrintValue<T>(a[i]);
        printf("  ");
        if (i % 10 == 9) printf("\n");
    }
    printf("~~~\n");

}



/******************************************************************
 * Compare two arrays
 ******************************************************************/

template <typename T>
int CompareArray(T* first, T* second, int len)
{
    for (int i = 0; i < len; i++) {
        if (first[i] != second[i]) {
            printf("Difference: %d\n", i);
            PrintValue<T>(first[i]);
            printf(" != ");
            PrintValue<T>(second[i]);
            printf("\n");
            return 1;
        }
    }   

    printf("Exactly the same\n");       
    return 0;       
}


/******************************************************************
 * Timing
 ******************************************************************/

struct GPUTimer {

   	cudaEvent_t start;
	cudaEvent_t stop;

	GPUTimer()
	{
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
	}

	~GPUTimer()
	{
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

	void Start()
	{
		cudaEventRecord(start, 0);
	}

	void Stop()
	{
		cudaEventRecord(stop, 0);
	}

	float ElapsedTime()
	{
		float elapsed;
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsed, start, stop);
		return elapsed;
	}
};



} // namepspace util

} // namepspace Morgen
