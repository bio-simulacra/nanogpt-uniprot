#!/bin/bash

# Default number of processors per node
DEFAULT_NPROC=8

# Check if the user provided a number of processors
if [ $# -eq 0 ]; then
    echo "No number of processors specified. Using default: $DEFAULT_NPROC"
    NPROC=$DEFAULT_NPROC
else
    NPROC=$1
    echo "Using user-specified number of processors: $NPROC"
fi

# Run the torchrun command with the specified or default number of processors
torchrun --standalone --nproc_per_node=$NPROC src/nanogpt_uniprot/train_gpt2.py