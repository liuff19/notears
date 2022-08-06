#!/bin/bash
for s0 in {10,20,30}
do
    d=10
    sem_type="gp"
    graph_type="SF"
    python nonlinear_adp.py --s0 ${s0} --d ${d} --sem_type ${sem_type} --graph_type ${graph_type} 
done
