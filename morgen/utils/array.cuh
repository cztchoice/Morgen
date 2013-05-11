/*
 *   Simple array operations 
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



namespace Morgen {

namespace util {


/**
 * Randomize an array
 */
template <typename T>
void RandomizeArray(T* a, int len, T max = 65536) {
    srand(time(0));
    for (int i = 0; i < len; i++) {
        a[i] = (T) (rand() % max);   // rand() returns an integer
    }

}




/**
 * Print an array
 */
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



/**
 * Compare two arrays
 */

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


} // namepspace util

} // namepspace Morgen
