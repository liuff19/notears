#!/bin/bash
for seed in {0..10}
do
    for run_mode in {1,3}
    do
        for d in {10}
        do
            for multiplier in {2,3,4}
            do
                s0=$(($d*$multiplier))
                sem_type="gp"
                graph_type="ER"
                python nonlinear_w.py --s0 ${s0} --d ${d} --sem_type ${sem_type} --graph_type ${graph_type} --run_mode ${run_mode}
            done
        done
    done
done

for seed in {0..10}
do
    for run_mode in {1,3}
    do
        for d in {20,30,40}
        do
            for multiplier in {2,3,4}
            do
                batch_size=500
                s0=$(($d*$multiplier))
                sem_type="gp"
                graph_type="ER"
                python nonlinear_w.py --s0 ${s0} --d ${d} --sem_type ${sem_type} --graph_type ${graph_type} --run_mode ${run_mode} --batch_size ${batch_size}
            done
        done
    done
done