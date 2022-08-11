#!/bin/bash

for d in {10,20,30,40,50}
do
    for multiplier in {2,3,4}
    do
        for seed in {0..10}
        do
            for method in {'DAG-GNN','DARING','NoCurl'}
            do
                # for n in {1000}
                # do 
                n=1000
                s0=$(($d*$multiplier))
                sem_type="gp"
                graph_type="ER"
                python nonlinear_exp.py --s0 ${s0} --d ${d} --sem_type ${sem_type} --graph_type ${graph_type} --seed ${seed} --method ${method} --n ${n}
                # done 
            done
        done
    done
done


for d in {10,20,30,40,50}
do
    for multiplier in {2,3,4}
    do
        for seed in {0..10}
        do
            for method in {"DAG-GNN","DARING","NoCurl"}
            do
                # for n in {1000}
                # do 
                n=1000
                s0=$(($d*$multiplier))
                sem_type="gp"
                graph_type="SF"
                python nonlinear_exp.py --s0 ${s0} --d ${d} --sem_type ${sem_type} --graph_type ${graph_type} --seed ${seed} --method ${method} --n ${n}
                # done 
            done
        done
    done
done