#!/bin/bash

echo "Bash version ${BASH_VERSION}..."
for i in {2..10}
  do
     a=$((i*24))
     python src/evaluate.py BLAT_ECOLX_Ranganathan2015-2500 onehot --n_seeds=1 --n_threads=1 --n_train=$a
  done
