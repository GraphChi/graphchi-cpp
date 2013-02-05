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
    
    int wanted = atoi(argv[2]);
    size_t nout = 0;
    size_t nin = 0;
    size_t nonz = 0;
    size_t tot = 0;
    degree d;
    int j = 0;
    while(!feof(f)) {
        fread(&d, sizeof(degree), 1, f);
        nout += d.outdegree;
        nin += d.indegree;
        if (wanted == j) {
            std::cout << wanted << " indeg: " << d.indegree << " outdeg: " << d.outdegree << std::endl;
            break;
        }       
        j++;
    }
    std::cout << "Total in: " << nin << " total out: " << nout << std::endl;
    std::cout << "Non-singleton vertices: " << nonz << std::endl;
    std::cout << "Total vertices: " << tot << std::endl;
    
}