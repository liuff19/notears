#!/bin/bash
for multiplier in {2,3,4}
do
    for seed in {0..10}
    do
        for mode in {1,3}
        do
            n=1000
            d=10
            s0=$(($d*$multiplier))
            sem_type="gp"
            graph_type="ER"
            python nonlinear_w.py --s0 ${s0} --d ${d} --sem_type ${sem_type} --graph_type ${graph_type} --seed ${seed} --batch_size ${batch} --run_mode ${mode} --n ${n}
        done
    done
done

for multiplier in {2,3,4}
do
    for seed in {0..10}
    do
        for mode in {1,3}
        do
            n=1000
            d=10
            s0=$(($d*$multiplier))
            sem_type="gp"
            graph_type="SF"
            python nonlinear_w.py --s0 ${s0} --d ${d} --sem_type ${sem_type} --graph_type ${graph_type} --seed ${seed} --batch_size ${batch} --run_mode ${mode} --n ${n}
        done
    done
done