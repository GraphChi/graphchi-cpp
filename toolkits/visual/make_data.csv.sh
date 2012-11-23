#!/bin/bash

GRAPHCHI_ROOT=~/graphchi/
CURPATH=`pwd`
NUM_EDGES=0
FILENAME=""
SEEDS=""
HOPS=2

PROGRAM_OPTIONS="f:h:n:s:"
HELP_STRING="-f Graph_file [-h hops] [-n edge_num] [-s seeds]\n For example: `basename $0` -f GRAPH_0.TSV -n 1000 -s 12,36,112";

if ( ! getopts $PROGRAM_OPTIONS opt); then
  echo "Usage: `basename $0` options $HELP_STRING"
  exit $E_OPTERROR;
fi

while getopts $PROGRAM_OPTIONS opt; 
do
  case $opt in
    f) FILENAME=$OPTARG;;
    n) NUM_EDGES=$OPTARG;;
    s) SEEDS=$OPTARG;;
    h) HOPS=$OPTARG;;
  esac
done

if [ $NUM_EDGES -gt 2000 ]; then
  echo "You should specify number of edges using -n XX, where XX is the number of edges to be subtructed (default 1000, max 2000)"
  exit
fi

if [ ! -f $FILENAME ]; then
  echo "Failed to open file: $FILENAME. Specify filename using -f XXXX command."
  exit
fi

#grab edges starting from a set of seeds
if [ ! -z $SEEDS ]; then
  cd $GRAPHCHI_ROOT
  ./toolkits/graph_analytics/subgraph --training=$FILENAME --seeds=$SEEDS --hops=$HOPS --edges=$NUM_EDGES --quiet=1
  mv $FILENAME.out  $FILENAME.$NUM_EDGES
else # else grab edges from the head of the file
  head -n $NUM_EDGES $FILENAME > $FILENAME.$NUM_EDGES
fi
exit

#translate the node ids to consecutive id 1..n
echo "$FILENAME.$NUM_EDGES" > $GRAPHCHI_ROOT/a
cd $GRAPHCHI_ROOT
./toolkits/parsers/consecutive_matrix_market --file_list=a --binary=1 --single_domain=1 
mv auser.reverse.map.text $CURPATH
mv $FILENAME.$NUM_EDGES.out $CURPATH
cd $CURPATH
exit

nodes_num=0;
declare -A names;
while read i
do
   names[`echo $i|awk '{print $1}'`]=`echo $i|awk '{print $2}'`;
   (( nodes_num++ ))
done < auser.reverse.map.text

#echo "${!names[*]}"

echo "source,target" > graph${NUM_EDGES}.csv
edges_num=0;
while read i
do
  echo "${names[`echo $i|awk '{print $1}'`]},${names[`echo $i|awk '{print $2}'`]}">> graph${NUM_EDGES}.csv
  (( edges_num++ ))
done < $FILENAME.$NUM_EDGES.out  

#write statistics to file
echo "[graph][$FILENAME]" > settingA
echo "[nodes][$nodes_num]" >> settingA
echo "[edges][$edges_num]" >> settingA

#remove intermediate files
rm -f $GRAPHCHI_ROOT/$FILENAME.$NUM_EDGES 

echo "scp bickson@thrust.ml.cmu.edu:~/tmp/graph${NUM_EDGES}.csv ~/Downloads/visualization_demo/gv_demo/"
echo "scp bickson@thrust.ml.cmu.edu:~/tmp/settingA ~/Downloads/visualization_demo/gv_demo/"
