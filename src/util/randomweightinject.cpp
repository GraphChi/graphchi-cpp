//
//  randomweightinject.cpp
//  graphchi_xcode
//
//  Created by Aapo Kyrola on 9/20/13.
//
//
// Takes an adge list and adds random weights to it

 
#include <cstdlib>
#include <iostream>
#include <assert.h>

typedef unsigned int vid_t;

// Removes \n from the end of line
inline void FIXLINE(char * s) {
    int len = (int) strlen(s)-1;
    if(s[len] == '\n') s[len] = 0;
}

int main(int argc, const char ** argv) {
    FILE * inf = fopen(argv[1], "r");
    std::string type(argv[2]);
    
    char outname[1024];
    sprintf(outname, "%s_weighted", argv[1]);
    std::cout << "Output: " << outname << std::endl;
    
    FILE * outf = fopen(outname, "w");
    
    if (type == "edgelist" ) {
        size_t bytesread = 0;
        size_t linenum = 0;
        
        assert(inf != NULL);
        
        char s[1024];
        while(fgets(s, 1024, inf) != NULL) {
            linenum++;
            if (linenum % 10000000 == 0) {
                std::cout << "Read " << linenum << " lines, " << bytesread / 1024 / 1024.  << " MB" << std::endl;
            }
            FIXLINE(s);
            bytesread += strlen(s);
            if (s[0] == '#') continue; // Comment
            if (s[0] == '%') continue; // Comment
           
            char delims[] = "\t, ";
            char * t;
            
            t = strtok(s, delims);
            vid_t from = atoi(t);
            t = strtok(NULL, delims);
            vid_t to = atoi(t);
            fprintf(outf,"%d %d %d\n", from, to, random() % 100000000);
        }
    } else if (type == "adjlist") {
        int maxlen = 100000000;
        char * s = (char*) malloc(maxlen);
        
        size_t bytesread = 0;
        
        char delims[] = " \t";
        size_t linenum = 0;
        size_t lastlog = 0;
        /*** PHASE 1 - count ***/
        while(fgets(s, maxlen, inf) != NULL) {
            linenum++;
            if (bytesread - lastlog >= 500000000) {
                std::cout << "Read " << linenum << " lines, " << bytesread / 1024 / 1024.  << " MB" << std::endl;
                lastlog = bytesread;
            }
            FIXLINE(s);
            bytesread += strlen(s);
            
            if (s[0] == '#') continue; // Comment
            if (s[0] == '%') continue; // Comment
            char * t = strtok(s, delims);
            vid_t from = atoi(t);
            t = strtok(NULL,delims);
            if (t != NULL) {
                vid_t num = atoi(t);
                vid_t i = 0;
                while((t = strtok(NULL,delims)) != NULL) {
                    vid_t to = atoi(t);
                    if (from != to)
                        fprintf(outf,"%d %d %d\n", from, to, random() % 100000000);
                    i++;
                }
                if (num != i) {
                    std::cerr << "Mismatch when reading adjacency list: " << num << " != " << i << " s: " << std::string(s)
                    << " on line: " << linenum << std::endl;
                    continue;
                }
            }
        }
        free(s);

    }
    
    fclose(outf);
}