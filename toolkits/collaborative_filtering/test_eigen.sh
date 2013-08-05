#!/bin/sh
if [ ! -d "../../src/Eigen/" ]; then
   echo "********************************************************************************"
   echo "Failed to find Eigen linear algebra package!"
   echo "Please follow step 3 of the instructions here: "
   echo "http://bickson.blogspot.co.il/2012/12/collaborative-filtering-with-graphchi.html"
   echo "********************************************************************************"
   exit 1;
else
   echo "Found Eigen linear algebra package!"
   exit 0;
fi
