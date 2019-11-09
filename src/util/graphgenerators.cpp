//
//  graphgenerators.cpp
//  graphchi_xcode
//
//  Created by Aapo Kyrola on 9/16/13.
//
//
 #include <stdlib.h>
#include <cstdio>
#include <sys/time.h>
#include <time.h>
#include <cmath>
#include <vector>
#include <iostream>
int idcounter = 0;

struct Node {
    std::vector<Node *> children;
    int id;
    
    Node(int id) : id(id) {}
    
    void branch(int mydepth, int maxdepth, int d, int n) {
        if (mydepth > maxdepth) return;
        while (children.size() < d) {
            children.push_back(new Node(idcounter++));
        }
        for(int i=0; i<children.size(); i++) {
            children[i]->branch(mydepth + 1, maxdepth, d, n);
        }
    }
    
    void output(FILE * f) {
        for(int i=0; i<children.size(); i++) {
            fprintf(f, "%d %d\n", id, children[i]->id);
            children[i]->output(f);
        }
    }
};

struct Tree {
    int d;
    Node * root;
    Tree(int d, int n) : d(d) {
        int depth = (int) (1 + log(n)/log(d));
        std::cout << "depth=" << depth<< std::endl;
        root = new Node(idcounter++);
        root->branch(1, depth, d, n);
    }
    
    void output(FILE * f) {
        root->output(f);
    }
};

int main(int argc, const char ** argv) {
    if (argc < 3) {
        printf("Usage: generate type n\n");
        return 1;
    }
    
    std::string type = argv[1];
    int n = atoi(argv[2]);
    
    
    char filename[1024];
    sprintf(filename, "%s_%d.edgelist", type.c_str(), n);
    FILE * f = fopen(filename, "w");
    
    if (type == "chain") {
        for(int x=0; x<n - 1; x++) {
            fprintf(f, "%d %d\n", x, x + 1);
        }
    }
    
    if (type == "grid") {
        for(int x=0; x<n; x++) {
            for(int y=0; y<n; y++) {
                int vid = y * n + x;
                if (x < n - 1)
                    fprintf(f, "%d %d\n", vid, vid + 1);
                if (y < n -1)
                    fprintf(f, "%d %d\n", vid, vid + n);
                
            }
        }
    }
    
    
    if (type == "crossgrid") {
        for(int x=0; x<n; x++) {
            for(int y=0; y<n; y++) {
                int vid = y * n + x;
                if (x < n - 1)
                    fprintf(f, "%d %d\n", vid, vid + 1);
                if (y < n - 1)
                    fprintf(f, "%d %d\n", vid, vid + n);
                if (x < n - 1 && y < n -1) {
                    // down and right
                    fprintf(f, "%d %d\n", vid, vid + n + 1);
                }
                if (x < n - 1 && y > 0) {
                    // up and right
                    fprintf(f, "%d %d\n", vid, vid -n + 1);
                }
            }
        }
    }
    
    if (type == "cubegrid") {
        for(int x=0; x<n; x++) {
            for(int y=0; y<n; y++) {
                for(int z=0; z<n; z++) {
                    int vid =  z * n * n + y * n + x;
                    if (x < n - 1)
                        fprintf(f, "%d %d\n", vid, vid + 1);
                    if (y < n -1)
                        fprintf(f, "%d %d\n", vid, vid + n);
                    if (z < n - 1)
                        fprintf(f, "%d %d\n", vid, vid + n * n);
                }
                
            }
        }
    }
    if (type == "quadgrid") {
        for(int x=0; x<n; x++) {
            for(int y=0; y<n; y++) {
                for(int z=0; z<n; z++) {
                    for(int w=0; w<n; w++) {
                        int vid =  w * n * n * n + z * n * n + y * n + x;
                        if (x < n - 1)
                            fprintf(f, "%d %d\n", vid, vid + 1);
                        if (y < n -1)
                            fprintf(f, "%d %d\n", vid, vid + n);
                        if (z < n - 1)
                            fprintf(f, "%d %d\n", vid, vid + n * n);
                        if (w < n - 1)
                            fprintf(f, "%d %d\n", vid, vid + n * n * n);
                        
                    }
                }
                
            }
        }
    }
    
    if (type == "tree2") {
        Tree t(2, n);
        t.output(f);
    }
    
    if (type == "tree3") {
        Tree t(3, n);
        t.output(f);
    }
    
    if (type == "tree4") {
        Tree t(4, n);
        t.output(f);
    }
    
    
    if (type == "tree5") {
        Tree t(5, n);
        t.output(f);
    }
    
    if (type == "tree50") {
        Tree t(50, n);
        t.output(f);
    }
    
    fclose(f);
}