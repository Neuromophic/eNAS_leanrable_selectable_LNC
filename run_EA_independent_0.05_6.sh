#!/bin/bash

#SBATCH --partition=multiple
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=1125mb
#SBATCH --time=72:00:00
#SBATCH --job-name=EAI0.05
#SBATCH --error=%x.%j.err
#SBATCH --output=%x.%j.out
#SBATCH --mail-user=hzhao@teco.edu

python3 experiment_EA.py --DATASET 08 --SEED 00 --e_train 0.05 --projectname VariationAware_EA &
python3 experiment_EA.py --DATASET 08 --SEED 01 --e_train 0.05 --projectname VariationAware_EA &
python3 experiment_EA.py --DATASET 08 --SEED 02 --e_train 0.05 --projectname VariationAware_EA &
python3 experiment_EA.py --DATASET 08 --SEED 03 --e_train 0.05 --projectname VariationAware_EA &
python3 experiment_EA.py --DATASET 08 --SEED 04 --e_train 0.05 --projectname VariationAware_EA &
python3 experiment_EA.py --DATASET 08 --SEED 05 --e_train 0.05 --projectname VariationAware_EA &
python3 experiment_EA.py --DATASET 08 --SEED 06 --e_train 0.05 --projectname VariationAware_EA &
python3 experiment_EA.py --DATASET 08 --SEED 07 --e_train 0.05 --projectname VariationAware_EA &
python3 experiment_EA.py --DATASET 08 --SEED 08 --e_train 0.05 --projectname VariationAware_EA &
python3 experiment_EA.py --DATASET 08 --SEED 09 --e_train 0.05 --projectname VariationAware_EA &
python3 experiment_EA.py --DATASET 09 --SEED 00 --e_train 0.05 --projectname VariationAware_EA &
python3 experiment_EA.py --DATASET 09 --SEED 01 --e_train 0.05 --projectname VariationAware_EA &
python3 experiment_EA.py --DATASET 09 --SEED 02 --e_train 0.05 --projectname VariationAware_EA &
python3 experiment_EA.py --DATASET 09 --SEED 03 --e_train 0.05 --projectname VariationAware_EA &
python3 experiment_EA.py --DATASET 09 --SEED 04 --e_train 0.05 --projectname VariationAware_EA &
python3 experiment_EA.py --DATASET 09 --SEED 05 --e_train 0.05 --projectname VariationAware_EA &

wait
