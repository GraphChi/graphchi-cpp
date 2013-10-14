//
//  mfrecommender.cpp
//  graphchi_xcode
//
//  Created by Aapo Kyrola on 10/14/13.
//
//

#include "graphchi_basic_includes.hpp"

using namespace graphchi;

#define NLATENT 20

typedef double factor_t[NLATENT];

factor_t * users;
factor_t * movies;
int num_users, num_movies;
int testn;

FILE * outputf;

double predict(int userid, int movieid) {
    factor_t & uf = users[userid];
    factor_t & mf = movies[movieid];
    
    double x  = 0;
    for(int j=0; j<NLATENT; j++) {
        x += uf[j] * mf[j];
    }
    return x;
}

double predict(int userid, int clusterid, factor_t * centroids) {
    factor_t & uf = users[userid];
    factor_t & cf = centroids[clusterid];
    
    double x  = 0;
    for(int j=0; j<NLATENT; j++) {
        x += uf[j] * cf[j];
    }
    return x;
}

double distance_sqr(int movieid, int centroidid, factor_t * centroids) {
    factor_t & mf = movies[movieid];
    factor_t & cf = centroids[centroidid];
    double x  = 0;
    for(int j=0; j<NLATENT; j++) {
        x += (cf[j] - mf[j]) * (cf[j] - mf[j]);
    }
    return x;

}

double getbestdot(int userid) {
    double bestdot = -1e30;
    for(int m=0; m<num_movies; m++) {
        double predrating = predict(userid, m);
        if (predrating > bestdot) bestdot = predrating;
    }
    return bestdot;
}

void brute() {
    for(int userid = 0; userid < testn; ++userid) {
        double bestdot = getbestdot(userid);
        fprintf(outputf, "%d,%lf\n", userid, bestdot);
        if (userid % 1000 == 0) std::cout << "Pred: " << userid << std::endl;
    }
}

void kmeansmethod(metrics &m) {
    m.start_time("kmeans");
    int kmeans_iters = get_option_int("kmeansiters", 100);
    int k = get_option_int("clusters", num_movies / 200);
    
    std::cout << "Running k-means for " << k << " clusters" << std::endl;
    
    factor_t * centroids = new factor_t[k];
    
    // Find factor range
    double minf=1e30, maxf = -1e30;
    double * vals = (double*)movies;
    for(size_t i=0; i < NLATENT * num_movies; i++) {
        minf = std::min(vals[i], minf);
        maxf = std::max(vals[i], maxf);
    }
    
    std::cout << "Factor range:" << minf << " --- " << maxf << std::endl;
    
    // Random starting points
    for(int i=0; i<k; i++) {
        for(int j=0; j<NLATENT; j++) {
            centroids[i][j] = (random() % 1000) * 0.001 * (maxf - minf) + minf;
        }
    }
    
    std::cout << "STARTING K-MEANS" << std::endl;
    
    int * assignments = new int[num_movies];
    
    for(int iter=0; iter < kmeans_iters; ++iter) {
        std::cout << "K-means iter" << iter << std::endl;
        // Assign centroids
        for(int j=0; j<num_movies; j++) {
            double mindist = 1e30;
            int newcentroid = 0;
            for(int c=0; c<k; c++) {
                double centroiddist = distance_sqr(j, c, centroids);
                if (centroiddist < mindist) {
                    newcentroid = c;
                    mindist = centroiddist;
                }
            }
            assignments[j] = newcentroid;
        }
        // New centroids
        int totalassgn = 0; // sanity
        for(int c=0; c<k; c++) {
            for(int j=0; j<NLATENT; j++) {
                centroids[c][j] =  0.0;
            }
            int numass = 0;
            
            for(int i=0; i<num_movies; i++) {
                if (assignments[i] == c) {
                    totalassgn++;
                    numass++;
                    for(int j=0; j<NLATENT; j++) {
                        centroids[c][j] += movies[i][j];
                    }
                }
            }
            if (numass > 0) {
                for(int j=0; j<NLATENT; j++) {
                    centroids[c][j] /= numass;
                }
            } else {
                // Randomizee again
                for(int j=0; j<NLATENT; j++) {
                    centroids[c][j] = (random() % 1000) * 0.001 * (maxf - minf) + minf;
                }
            }
        }
        assert(totalassgn == num_movies);
    }
    
    m.stop_time("kmeans");
    
    std::vector< std::vector<int> > clustersets;
    for(int c=0; c<k; c++) {
        clustersets.push_back(std::vector<int>());
        for(int j=0; j<num_movies; j++) {
            if (assignments[j] == c) clustersets[c].push_back(j);
        }
    }
    
    
    m.start_time("kmeans-based-recs");
    
    std::vector<double> bests;
    bests.reserve(testn);
    
    for(int userid = 0; userid < testn; ++userid) {
        double bestclusterdot = -1e30;
        int bestcluster = -1;
        for(int c=0; c<k; c++) {
            if (!clustersets[c].empty()) {
                double predrating = predict(userid, c, centroids);
                if (predrating > bestclusterdot) {
                    bestclusterdot = predrating;
                    bestcluster = c;
                }
            }
        }
        assert(bestcluster >= 0);
        
        /* Then look into the cluster */
        std::vector<int> & clusteritems = clustersets[bestcluster];
        assert(clusteritems.size() > 0);
        
        double bestdot = -1e30;
        for(int j=0; j<clusteritems.size(); j++) {
            double d = predict(userid, clusteritems[j]);
            if (d > bestdot) bestdot = d;
        }
        
        fprintf(outputf, "%d,%lf\n", userid, bestdot);
        bests[userid] = bestdot;
        if (userid % 1000 == 0) std::cout << "Pred: " << userid << std::endl;
    }
    
    m.stop_time("kmeans-based-recs");
    
    std::cout << "Compute error" << std::endl;
    double err = 0;
    for(int userid=0; userid<testn; ++userid) {
        double optimal = getbestdot(userid);
        double approx = bests[userid];
        err += optimal - approx;
    }
    std::cout << "Average error: " << (err / testn) << std::endl;
 }

int main(int argc, const char ** argv) {
    graphchi_init(argc, argv);
    
    metrics m("rec");
    std::string basefilename = get_option_string("file");
    num_users = get_option_int("users");
    num_movies = get_option_int("movies");
    testn = get_option_int("testusers", 10000);
    
    std::string mode = get_option_string("mode");
    
    std::string outputfile = basefilename + "." + mode;
    outputf = fopen(outputfile.c_str(), "w");
    factor_t * allfactors;
    std::string vfilename = filename_vertex_data<factor_t>(basefilename);
    int f = open(vfilename.c_str(), O_RDONLY);
    std::cout << vfilename << std::endl;
    assert(f >= 0);
    size_t size = readfull(f, &allfactors);
    std::cout << "size: " << size << std::endl;
    std::cout << "expect: " << (num_users + num_movies) * sizeof(factor_t) << std::endl;
    assert(size == (num_users + num_movies) * sizeof(factor_t));
    
    users = allfactors;
    movies = &allfactors[num_users];
    
    m.start_time("recommend." + mode);
    if (mode == "all") {
        brute();
    } else if (mode == "kmeans") {
        kmeansmethod(m);
    }
    m.stop_time("recommend." + mode);

    std::cout << "Done" << std::endl;
    fclose(outputf);
    free(allfactors);
    metrics_report(m);

    return 0;
}
