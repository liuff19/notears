#!/bin/bash
for d in {10,15}
do
    for multiplier in {2,3,4}
    do
        for seed in {0,5,11,17,21}
        do
            s0=$(($d*$multiplier))
            sem_type="gp-add"
            graph_type="ER"
            python nonlinear_adp.py --s0 ${s0} --d ${d} --sem_type ${sem_type} --graph_type ${graph_type} --seed ${seed}
        done
    done
done


for d in {10,15}
do
    for multiplier in {2,3,4}
    do
        for seed in {0,5,11,17,21}
        do
            s0=$(($d*$multiplier))
            sem_type="gp-add"
            graph_type="SF"
            python nonlinear_adp.py --s0 ${s0} --d ${d} --sem_type ${sem_type} --graph_type ${graph_type} --seed ${seed}
        done
    done
done