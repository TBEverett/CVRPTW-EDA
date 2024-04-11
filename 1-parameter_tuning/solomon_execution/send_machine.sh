#!/bin/bash

#Codigo para 

pass=$2

if [ $1 = "ton" ]
then
    path="tbarros@ssh2.inf.utfsm.cl:/home/tbarros/Tesis"
fi

if [ $1 = "mach" ]
then 
    path="elizabeth@s2.labia.inf.utfsm.cl:/home/elizabeth/Tomas/Tesis"
fi
echo "Starting Transfer"
sshpass -p $pass scp -r baseTuning $path
sshpass -p $pass scp -r execute.sh $path
sshpass -p $pass scp -r requirements.txt $path
sshpass -p $pass scp -r send_machine.sh $path
sshpass -p $pass scp -r Homberger $path
sshpass -p $pass scp -r solomon $path
sshpass -p $pass scp -r run_per_instance.py $path
sshpass -p $pass scp -r README.md $path
echo "Transfer Done"