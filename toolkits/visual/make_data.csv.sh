#!/bin/bash

cd ../../
GRAPHCHI_ROOT=`pwd`
cd -
CURPATH=`pwd`
NUM_EDGES=0
FILENAME=""
SEEDS=""
HOPS=2
NODES=0;
EDGES=0;

PROGRAM_OPTIONS="f:h:n:o:s:"
HELP_STRING="-f Graph_file [-h hops] [-n edge_num] [-o nodes] [-s seeds]\n For example: `basename $0` -f GRAPH_0.TSV -n 1000 -s 12,36,112";

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
    o) NODES=$OPTARG;;
    \?) echo "Usage: `basename $0` options $HELP_STRING";;
  esac
done

if [ $NUM_EDGES -gt 2000 ]; then
  echo "You should specify number of edges using -n XX, where XX is the number of edges to be subtructed (default 1000, max 2000)"
  exit
fi

OPIOTNAL_ARG=""

#check that input file exists
if [ ! -f $FILENAME ]; then
  echo "Failed to open file: $FILENAME. Specify filename using -f XXXX command."
  exit
elif [ ! -z $SEEDS ]; then # check that the input file has the right format for cutting a subgraph from
  head -n 1 $FILENAME | grep -q "^%%MatrixMarket";
  if [ $? -eq 1 ]; then #matrix market header was not found
    if [ $NODES -eq 0 ]; then
      echo "For graph input file which is not in matrix market sparse format, you need to use the -o command to specify an upper limit on the number of nod ids"
      exit
    else
      EDGES=`wc -l $FILENAME`;
      if [ -z `head -n 1 $FILENAME | cut -f 3` ]; then
         OPIOTNAL_ARG="--tokens_per_row=2"
      fi
    fi
  fi
fi

#grab edges starting from a set of seeds
if [ ! -z $SEEDS ]; then
  cd $GRAPHCHI_ROOT
  ./toolkits/graph_analytics/subgraph --training=$FILENAME --seeds=$SEEDS --hops=$HOPS --edges=$NUM_EDGES --nodes=$NODES --orig_edges=$EDGES --quiet=1 $OPIOTNAL_ARG
  mv $FILENAME.out  $FILENAME.$NUM_EDGES
else # else grab edges from the head of the file
  head -n $NUM_EDGES $FILENAME > $FILENAME.$NUM_EDGES
fi

#translate the node ids to consecutive id 1..n
echo "$FILENAME.$NUM_EDGES" > $GRAPHCHI_ROOT/a
cd $GRAPHCHI_ROOT
./toolkits/parsers/consecutive_matrix_market --file_list=a --binary=1 --single_domain=1 
mv auser.reverse.map.text $CURPATH
#mv $FILENAME.$NUM_EDGES.out $CURPATH
cd $CURPATH

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
echo "[hops][$HOPS]" >> settingA
echo "[seeds][$SEEDS]" >> settingA
echo "[creation][`date`]" >> settingA
echo "[user][$USER]" >> settingA
#remove intermediate files
rm -f $GRAPHCHI_ROOT/$FILENAME.$NUM_EDGES 
mv graph${NUM_EDGES}.csv graph1000.csv # ugly, need to find a way to pass graph name to UI
echo "scp bickson@thrust.ml.cmu.edu:~/tmp/graph${NUM_EDGES}.csv bickson@thrust.ml.cmu.edu:~/tmp/settingA ~/Downloads/visualization_demo/gv_demo/"
