/*
 *   Display the out degree histogram
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
#include <morgen/graph/gen/tiny.cuh>
#include <cstdlib>
#include <stdio.h>
#include <iostream>


using namespace Morgen;
using namespace CommandLineProcessing;
using namespace std;


/********************************************************************************
 * Globals
 ********************************************************************************/

string   g_input_graph      =  "";     


/**********************************************************************************
 * Init parameters
 **********************************************************************************/

void Init(int argc, char** argv) {

    ArgvParser cmd;

    // help option
    cmd.setHelpOption("help", "h", "help menu");

    // options with value
    cmd.defineOption("input", "the path of input graph",
                     ArgvParser::OptionRequiresValue); 

    ArgvParser::ParserResults result = cmd.parse(argc, argv);


    if (result != ArgvParser::NoParserError) {
        cout << cmd.parseErrorDescription(result)
             << "  (type -h for help) \n";
        exit(1);
    }

    
    if (cmd.foundOption("input")) {
        g_input_graph  = cmd.optionValue("input");
    }

}





/*************************************************************************************
 * Main
 *************************************************************************************/

int main(int argc, char** argv) 
{

    Init(argc, argv);

    Morgen::graph::CsrGraph<int, int> g;

    // generate g a tiny graph
    Morgen::graph::gen::TinyGraph<int, int>(g);
    

    g.Display();
    g.DisplayOutDegree();


    return 0;
}
