
/** 
  * Simple tool for creating input for streaming graph demos.
  * An edgelist is read and two files are created: base-graph and
  * streaming input file. Streaming input is shuffled.
  * NOTE: This is unsupported code and requires plenty of memory.
  */
  
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>

#include <string.h>
#include <string>
#include <errno.h>
#include <algorithm>
#include <iterator>

struct edge {
    unsigned int from;
    unsigned int to;
};


// Removes \n from the end of line
void FIXLINE(char * s) {
    int len = (int) strlen(s)-1; 	  
    if(s[len] == '\n') s[len] = 0;
}
    
int main(int argc, const char ** argv) {
    if (argc != 3) {
        std::cout << "Usage: [inputfile-edgelist] [stream-edges-per-base-edges] [max-base-id]" << std::endl;
    }
    const char * input = argv[1];
    int stream_edges_per_base_edges = atoi(argv[2]);
    int maxbaseid = atoi(argv[3]);
    std::cout << "Processing: " << input << std::endl;
    
     FILE * inf = fopen(input, "r");
        
    std::vector<edge> base_edges;
    std::vector<edge> stream_edges;
    base_edges.reserve(1e6);
    stream_edges.reserve(1e6);

    if (inf == NULL) {
        std::cout << "Could not load :" << input << " error: " << strerror(errno) << std::endl;
    }
    assert(inf != NULL);
    
     std::cout << "Reading in edge list format!" << std::endl;
    char s[1024];
    while(fgets(s, 1024, inf) != NULL) {
        FIXLINE(s);
        if (s[0] == '#') continue; // Comment
        if (s[0] == '%') continue; // Comment
        char delims[] = "\t ";	
        char * t;
        t = strtok(s, delims);
        edge e;
        e.from = atoi(t);
        t = strtok(NULL, delims);
        e.to = atoi(t);
        if (std::rand() % stream_edges_per_base_edges == 0 && e.from <= maxbaseid && e.to <= maxbaseid) base_edges.push_back(e);
        else stream_edges.push_back(e);
    }
    fclose(inf);
    
    std::cout << "Number of edges in base: " << base_edges.size() << std::endl;
    std::cout << "Number of edges to stream: " << stream_edges.size() << std::endl;
    
    std::string base_file_name = std::string(input) + "_base";
    std::string stream_file_name = std::string(input) + "_stream";
    
    FILE * basef = fopen(base_file_name.c_str(), "w");
    
    for(std::vector<edge>::iterator it=base_edges.begin();
                it != base_edges.end(); ++it) {
        fprintf(basef, "%u %u\n", it->from, it->to);           
    }
    
    fclose(basef);
    
    /* Shuffle */
    std::random_shuffle(stream_edges.begin(), stream_edges.end());
 
    
    FILE * strmf = fopen(stream_file_name.c_str(), "w");
    for(std::vector<edge>::iterator it=stream_edges.begin();
                it != stream_edges.end(); ++it) {
        fprintf(strmf, "%u %u\n", it->from, it->to);           
    }
    fclose(strmf);
    return 0;
}   
