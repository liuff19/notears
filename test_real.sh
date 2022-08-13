#!/bin/bash


# mode=1
# batch=853
# python nonlinear_w.py --run_mode ${mode} --batch_size ${batch}

mode=3
batch=853
iter_mod=5
clip=1000
data='sachs_full'
python nonlinear_w.py --run_mode ${mode} --batch_size ${batch} --iter_mod ${iter_mod} --clip ${clip} --data_type ${data}
