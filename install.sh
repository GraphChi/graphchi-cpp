#!/bin/bash 
#script for installing graphchi cf toolbox
#written by Danny Bickson, CMU
EIGEN_FILE=3.1.3.tar.bz2
EIGEN_DIST=http://bitbucket.org/eigen/eigen/get/$EIGEN_FILE

test -z `which wget`
if [ $? -eq 1 ]; then
  rm -fR $EIGEN_DIST
  if [ "$(uname -o)" == "Cygwin" ]; then
    # '--no-check-certificate' to silence complaint about bitbucket.org's certificate
    wget --no-check-certificate --max-redirect 20 $EIGEN_DIST
  else
    wget --max-redirect 20 $EIGEN_DIST
  fi
  if [ $? -ne 0 ]; then
     echo "Failed to download file"
     echo "Please download manually the file $EIGEN_DIST to the root GraphChi folder"
     exit 1
  fi
else
  test -z `which curl` 
  if [ $? -eq 1 ]; then
    rm -fR $EIGEN_DIST
    curl -o $EIGEN_FILE -L $EIGEN_DIST
    if [ $? -ne 0 ]; then
     echo "Failed to download file"
     echo "Please download manually the file $EIGEN_DIST to the root GraphChi folder"
     exit 1
    fi
  else
     echo "Failed to find wget or curl"
     echo "Please download manually the file $EIGEN_DIST to the root GraphChi folder"
     exit 1
  fi
fi
rm -f eigen-eigen-*
tar -xjf $EIGEN_FILE
    if [ $? -ne 0 ]; then
     echo "Failed to extract eigen files"
     echo "Please download manually the file $EIGEN_DIST to the root GraphChi folder"
     exit 1
    fi

rm -fR ./src/Eigen
mv eigen-eigen-*/Eigen ./src
cd toolkits/collaborative_filtering
make 
cd ../../
