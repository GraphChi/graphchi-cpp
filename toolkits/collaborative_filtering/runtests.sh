#!/bin/bash
export GRAPHCHI_ROOT=$PWD/../../

stdoutfname=$PWD/stdout.log
rm -f $stdoutfname 

echo | tee -a $stdoutfname
echo "Running application tests"| tee -a $stdoutfname
echo "========================="| tee -a $stdoutfname
echo "GraphChi collaborative filtering library"| tee -a $stdoutfname
somefailed=0
echo "---------ALS-------------"  | tee -a $stdoutfname
./als --unittest=1 --quiet=1 >> $stdoutfname 2>& 1
if [ $? -eq 0 ]; then
  echo "PASS TEST 1 (Alternating least squares)"| tee -a $stdoutfname
else
  somefailed=1
  echo "FAIL ./aks --unittest=1 (Alternating least squares)"| tee -a $stdoutfname
fi

echo "---------WALS-------------" | tee -a $stdoutfname
./wals --unittest=1 --quiet=1 >> $stdoutfname 2>& 1
if [ $? -eq 0 ]; then
  echo "PASS TEST 2 (Weighted alternating least squares)"| tee -a $stdoutfname
else
  somefailed=1
  echo "FAIL TEST 2 (Weighted Alternating least squares)"| tee -a $stdoutfname
fi

if [ $somefailed == 1 ]; then
  echo "**** FAILURE LOG **************" >> $stdoutfname
  echo `date` >> $stdoutfname
  echo `uname -a` >> $stdoutfname
  echo `echo $USER` >> $stdoutfname
  echo "Some of the tests failed".
  echo "Please email stdout.log to danny.bickson@gmail.com"
  echo "Thanks for helping improve GraphChi!"
fi




