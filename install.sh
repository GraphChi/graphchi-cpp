#!/bin/sh -e
#script for installing graphchi cf toolbox
#written by Danny Bickson, CMU
wget http://bitbucket.org/eigen/eigen/get/3.1.1.tar.bz2
tar -xjf 3.1.1.tar.bz2
mv eigen-eigen-*/Eigen ./src
cd toolkits/collaborative_filtering
make 
cd ../../
