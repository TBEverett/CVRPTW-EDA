#!/bin/bash

source ../../.env

toTune=$1
seed=$2
dir=$3
maxEvaluations=${PARAMILS_MAXEVALS}
algo=SolveVRP

respaldos=respaldos${algo}
mkdir ../${dir}/${respaldos}

for function in $( cat ${toTune} ); do
	#crear archivo scenario y archivo parametros
	scenario=../${dir}/scn/${function}.scn
        params=${function}.params
        instance=../${dir}/inst/${function}.inst
	echo ${function}.params
        outputTuner=ParamILS_A${algo}_F${function}_S${seed}.out
        echo "outputTuner=ParamILS_A${algo}_F${function}_S${seed}.out"
        echo "time ruby paramils2.3.8-source/param_ils_2_3_run.rb -numRun ${seed} -approach focused -userunlog 1 -validN 0 -pruning 0 -maxEvals ${maxEvaluations} -scenariofile ${scenario} > ../${dir}/${outputTuner}"
        time ruby ../${dir}/paramils2.3.8-source/param_ils_2_3_run.rb -numRun ${seed} -approach focused -userunlog 1 -validN 0 -pruning 0 -maxEvals ${maxEvaluations} -scenariofile ${scenario} > ../${dir}/${outputTuner}
        echo "mv out ../${dir}/${respaldos}/outA${algo}_F${function}_S${seed}"
        mv out ../${dir}/${respaldos}/outA${algo}_F${function}_S${seed}
	#done
        #mv ${scenario} ${params} ${instance} ${respaldos}
done

