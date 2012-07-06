INCFLAGS = -I/usr/local/include/ -I./src/

CPP = g++
CPPFLAGS = -O3 -g $(INCFLAGS) -m64 -fopenmp -Wall -Wno-strict-aliasing
DEBUGFLAGS = -g -ggdb $(INCFLAGS)
HEADERS=$(wildcard *.h**)


all: apps tests sharder_basic 
apps: example_apps/connectedcomponents example_apps/pagerank example_apps/pagerank_functional example_apps/communitydetection example_apps/trianglecounting
als: example_apps/matrix_factorization/als_edgefactors  example_apps/matrix_factorization/als_vertices_inmem
tests: tests/basic_smoketest tests/bulksync_functional_test


clean:
	@rm -rf bin/*

sharder_basic: src/preprocessing/sharder_basic.cpp $(HEADERS)
	$(CPP) $(CPPFLAGS) src/preprocessing/sharder_basic.cpp -o bin/sharder_basic

example_apps/% : example_apps/%.cpp $(HEADERS)
	@mkdir -p bin/$(@D)
	$(CPP) $(CPPFLAGS) -Iexample_apps/ $@.cpp -o bin/$@


myapps/% : myapps/%.cpp $(HEADERS)
	@mkdir -p bin/$(@D)
	$(CPP) $(CPPFLAGS) -Imyapps/ $@.cpp -o bin/$@

tests/%: src/tests/%.cpp $(HEADERS)
	@mkdir -p bin/$(@D)
	$(CPP) $(CPPFLAGS) src/$@.cpp -o bin/$@	


graphlab_als: example_apps/matrix_factorization/graphlab_gas/als_graphlab.cpp
	$(CPP) $(CPPFLAGS) example_apps/matrix_factorization/graphlab_gas/als_graphlab.cpp -o bin/graphlab_als

docs: */**
	doxygen conf/doxygen/doxygen.config


dist: */**
	@mkdir -p dist
	tar -czf dist/graphchi_src_vXX.tar.gz src example_apps conf docs README.txt graphchi_xcode Makefile 
	
	

	