#!/bin/bash
# script for visualizing a subgraph
# Written by Danny Bickson, CMU

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

PROGRAM_OPTIONS="f:h:n:o:r:s:"
HELP_STRING="-f Graph_file [-h hops] [-n edge_num] [-o nodes] [-s seeds] [-r multiple files regexp]\n For example: `basename $0` -f GRAPH_0.TSV -n 1000 -s 12,36,112";

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
r) FILENAMES="$OPTARG";; 
    \?) echo "Usage: `basename $0` options $HELP_STRING";;
  esac
done

if [ $NUM_EDGES -gt 2000 ]; then
  echo "You should specify number of edges using -n XX, where XX is the number of edges to be subtructed (default 1000, max 2000)"
  exit
fi

if [ ! -z "$FILENAMES" ]; then
  if [ ! -z $FILENAME ]; then
    echo "When using -r command to specify multiple files, you are not allowed to use the -f command"
    exit
  fi
  DIRNAME=`dirname "$FILENAMES"`
  WILDCARD=`basename "$FILENAMES"`
  FINDSTR="find \"$DIRNAME\" -name \"$WILDCARD\" | sort > input.files"
  FILENAMES=`eval $FINDSTR`
else
  FILENAMES=`basename $FILENAME`
  echo $FILENAMES > input.files
  DIRNAME=`dirname $FILENAME`
fi

OPIOTNAL_ARG=""

rm -fR file.txt
touch file.txt

file_num=0;
while read FILENAME 
do
  (( file_num++ ))
  echo "Going over file: $FILENAME file number: $file_num"
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
  echo "$DIRNAME/$FILENAME.$NUM_EDGES" > $GRAPHCHI_ROOT/a
  cd $GRAPHCHI_ROOT
  ./toolkits/parsers/consecutive_matrix_market --file_list=a --binary=1 --single_domain=1 
  rm -f ./toolkits/visual/auser.map.text
  mv auser.map.text ./toolkits/visual/
  cd $CURPATH

  if [ ! -f auser.map.text ]; then
    echo "Bug - missing file auser.map.text"
    exit 1
  fi

  echo "Going to go over `wc -l auser.map.text | awk '{print $1}'` lines of map file"
  nodes_num=0;
  declare -A names;
  while read i
  do
    names[`echo $i|awk '{print $2}'`]=`echo $i|awk '{print $1}'`;
    (( nodes_num++ ))
  done < auser.map.text

  #echo "${!names[*]}"
  if [ ! -f $FILENAME.$NUM_EDGES.out ]; then
     echo "Bug: failed to find $FILENAME.$NUM_EDGES.out"
     exit 1
  fi

  echo "source,target" > graph${NUM_EDGES}.csv
  edges_num=0;
  while read i
  do
    echo "${names[`echo $i|awk '{print $1}'`]},${names[`echo $i|awk '{print $2}'`]}">> graph${NUM_EDGES}.csv
    (( edges_num++ ))
  done < $FILENAME.$NUM_EDGES.out  

  #remove intermediate files
  rm -f $GRAPHCHI_ROOT/$FILENAME.$NUM_EDGES 

  mv graph${NUM_EDGES}.csv graph${NUM_EDGES}.$file_num.csv 
  echo "graph${NUM_EDGES}.$file_num.csv" >> file.txt
  echo "[filename][$FILENAME]" > graph${NUM_EDGES}.$file_num.csv.txt
  echo "[nodes][$nodes_num]" >> graph${NUM_EDGES}.$file_num.csv.txt
  echo "[edges][$edges_num]" >>graph${NUM_EDGES}.$file_num.csv.txt

done < input.files # for FILENAMES

#write statistics to file
echo "[directory][$DIRNAME]" > settingA
#echo "[nodes][$nodes_num]" >> settingA
#echo "[edges][$edges_num]" >> settingA
echo "[hops][$HOPS]" >> settingA
echo "[seeds][$SEEDS]" >> settingA
echo "[creation][`date`]" >> settingA
echo "[user][$USER]" >> settingA


echo "scp bickson@thrust.ml.cmu.edu:~/tmp/graph${NUM_EDGES}.csv bickson@thrust.ml.cmu.edu:~/tmp/settingA ~/Downloads/visualization_demo/gv_demo/"
