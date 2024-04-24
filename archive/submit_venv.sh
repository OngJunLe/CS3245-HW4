#!/bin/bash
#SBATCH --time=300
#SBATCH --job-name=cs3245_hw4
#SBATCH --partition=medium
#SBATCH --mem=128G

free -m

#python index.py
python -u index.py -i 'data/dataset.csv' -d 'data/test' -p 'data/test'