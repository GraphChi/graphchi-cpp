//
//  readdeg.cpp
//  graphchi_xcode
//
//  Created by Aapo Kyrola on 9/14/12.
//  Copyright 2012 __MyCompanyName__. All rights reserved.
//

#include <iostream>
#include <fstream>

struct degree {
    int indegree;
    int outdegree;
};

int main(int argc, const char ** argv) {
    FILE * f = fopen(argv[1], "r");
    
    size_t nout = 0;
    size_t nin = 0;
    degree d;
    while(!feof(f)) {
        fread(&d, sizeof(degree), 1, f);
        nout += d.outdegree;
        nin += d.indegree;
    }
    std::cout << "Total in: " << nin << " total out: " << nout << std::endl;
    
}