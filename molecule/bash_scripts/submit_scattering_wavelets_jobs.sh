#!/bin/bash

start_i=0
n_mols=10000
max_i=250000

while [ $start_i -lt $max_i ]; do
    end_i=$((start_i + n_mols))
    echo "start_i: $start_i, end_i: $end_i"
    sbatch run_scattering_inv_wavelet.sh --export=start_i
    start_i=$end_i
done
