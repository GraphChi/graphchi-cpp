#!/bin/bash

GRAPHCHI_ROOT=~/graphchi/
CURPATH=`pwd`
echo $#
if [ $# -eq 0 ]; then
  NUM=1000
  GRAPH=3
  FILENAME=GRAPHS_${GRAPH}.TSV
elif [ $# -eq 2 ]; then
  FILENAME=$1
  NUM=$2
else
  echo "Usage $0 [graph file name] [number of edges]"
  exit
fi

head -n $NUM $FILENAME > $GRAPHCHI_ROOT/$FILENAME.$NUM
echo "$FILENAME.$NUM" > $GRAPHCHI_ROOT/a
cd $GRAPHCHI_ROOT
./toolkits/parsers/consecutive_matrix_market --file_list=a --binary=1 --single_domain=1 
mv auser.reverse.map.text $CURPATH
mv $FILENAME.$NUM.out $CURPATH
cd $CURPATH

nodes_num=0;
declare -A names;
while read i
do
   names[`echo $i|awk '{print $1}'`]=`echo $i|awk '{print $2}'`;
   (( nodes_num++ ))
done < auser.reverse.map.text

#echo "${!names[*]}"

echo "source,target" > graph${NUM}.csv
edges_num=0;
while read i
do
  echo "${names[`echo $i|awk '{print $1}'`]},${names[`echo $i|awk '{print $2}'`]}">> graph${NUM}.csv
  (( edges_num++ ))
done < $FILENAME.$NUM.out  

#write statistics to file
echo "[graph][$FILENAME]" > settingA
echo "[nodes][$nodes_num]" >> settingA
echo "[edges][$edges_num]" >> settingA

#remove intermediate files
rm -f $GRAPHCHI_ROOT/$FILENAME.$NUM 

echo "scp bickson@thrust.ml.cmu.edu:~/tmp/graph${NUM}.csv ~/Downloads/visualization_demo/gv_demo/"
echo "scp bickson@thrust.ml.cmu.edu:~/tmp/settingA ~/Downloads/visualization_demo/gv_demo/"
