#!/bin/bash

for d in {20,30,40}
do
    for multiplier in {2,3,4}
    do
        for seed in {0..10}
        do
            for mode in {1,3}
            do
                for n in {1000,1500,2000}
                do 
                    batch=500
                    s0=$(($d*$multiplier))
                    sem_type="gp"
                    graph_type="ER"
                    python nonlinear_w.py --s0 ${s0} --d ${d} --sem_type ${sem_type} --graph_type ${graph_type} --seed ${seed} --batch_size ${batch} --run_mode ${mode} --n ${n}
                done 
            done
        done
    done
done


for d in {20,30,40}
do
    for multiplier in {2,3,4}
    do
        for seed in {0..10}
        do
            for mode in {1,3}
            do
                for n in {1000,1500,200}
                do 
                    batch=500
                    s0=$(($d*$multiplier))
                    sem_type="gp"
                    graph_type="SF"
                    python nonlinear_w.py --s0 ${s0} --d ${d} --sem_type ${sem_type} --graph_type ${graph_type} --seed ${seed} --batch_size ${batch} --run_mode ${mode} --n ${n}
                done 
            done
        done
    done
done