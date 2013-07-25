
/**
  * Simple utility to generate Erdos-Renyi G(n, p) graphs.
  * n = number of vertices, p = probability for each edge.
  * This requires O(n^2) time, so not good idea to define very large n.
  * Edge direction is randomized.
  */

#include <stdlib.h>
#include <cstdio>
#include <sys/time.h>
#include <time.h>

int main(int argc, const char ** argv) {
    if (argc < 4) {
        printf("Usage: erdosrenyi nameprefix n p\n");
        return 1;
    }   
    
    const char * nameprefix = argv[1];
    int n = atoi(argv[2]);
    double p = atof(argv[3]);
    
    printf("Number of vertices=%d, edge probability=%lf\n", n, p);
    
    char filename[1024];
    sprintf(filename, "erdosrenyi_%s_%d_%.4f.edgelist", nameprefix, n, p);
    
    timeval tt;
    gettimeofday(&tt, NULL);
    
    srandom(time(NULL) +  (int)tt.tv_usec);

    size_t K = 10000;
    
    FILE * f = fopen(filename, "w");
    
    size_t count = 0;
    size_t accepted = 0;
    
    size_t lim = (size_t) (p * K);
    
    for(int i=0; i < n; i++) {
        for(int j=i+1; j < n; j++) {
            count++;
            size_t r = random();
            if (r % K < lim) {
                accepted++;
                if (rand() % 2 == 0) {
                    fprintf(f, "%d\t%d\n", i, j);
                } else {
                    fprintf(f, "%d\t%d\n", j, i);
                }
            }
        }
    }
    
    fclose(f);
    
    printf("count: %lu, accepted: %lf\n", count, double(accepted)/double(count));
}