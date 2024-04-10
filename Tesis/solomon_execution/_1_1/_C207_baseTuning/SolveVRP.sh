#/bin/bash

source ../../.env

instance=$1
seed=$5
shift 5

while [ $# != 0 ]; do
    flag="$1"
    case "$flag" in
        -ps) if [ $# -gt 1 ]; then
              arg="$2"
              shift
              ps=$arg
            fi
            ;;
        -gs) if [ $# -gt 1 ]; then
              arg="$2"
              shift
              gs=$arg
            fi
            ;;
        -ne) if [ $# -gt 1 ]; then
              arg="$2"
              shift
              ne=$arg
            fi
            ;;
        -nc) if [ $# -gt 1 ]; then
              arg="$2"
              shift
              nc=$arg
            fi
            ;;
        -xi) if [ $# -gt 1 ]; then
              arg="$2"
              shift
              xi=$arg
            fi
            ;;
        *) echo "Unrecognized flag or argument: $flag"
            ;;
        esac
    shift
done

#ps=25 #Population Size
#gs=40 #Generation Size
#ne=0.1 #Number of elite solutions
#nc=0.1 #Number of neighbour solutions
#xi=0.5 #Upper Bound of Feasible Solutions
t=${SOLVEVRP_TIME_LIMIT} #Max Runtime

out=out.txt
ARGS="-i ${instance} -s ${seed} -ps ${ps} -gs ${gs} -ne ${ne} -nc ${nc} -xi ${xi} -t ${t}"
screen=salida
rm -rf ${screen}

echo "./python3 SolveVRP.py ${ARGS} > ${screen}"
python3 SolveVRP.py ${ARGS} > ${screen}

quality=`tail -1 ${screen} |awk -F ' ' '{print $1}'`

echo "Quality: ${quality}"
runlength=${quality}

solved="SAT"
runtime=0
best_sol=0

echo "Result for ParamILS: ${solved}, ${runtime}, ${runlength}, ${best_sol}, ${seed}"


