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

int main(int argc, char** argv) 
{
    
    ArgvParser cmd;

    cmd.setHelpOption("help", "h", "help menu");

    // options with value
    cmd.defineOption("stride", "stride of writing", ArgvParser::OptionRequiresValue); 
    cmd.defineOption("threads", "threads per block", ArgvParser::OptionRequiresValue);

    ArgvParser::ParserResults result = cmd.parse(argc, argv);

    if (result != ArgvParser::NoParserError) {
        cout << cmd.parseErrorDescription(result) << "  (type -h for help)" << endl;
        exit(1);
    }

    int stride = 1;
    int blocks = 1;
    int threads = 1024;

    if (cmd.foundOption("stride")) {
        const char* s = cmd.optionValue("stride").c_str();
        stride = atoi(s);
    }

    if (cmd.foundOption("threads")) {
        const char* s = cmd.optionValue("threads").c_str();
        threads = atoi(s);
    }


    return 0;
}
