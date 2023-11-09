#!/bin/bash

#SBATCH --partition=multiple
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=1125mb
#SBATCH --time=72:00:00
#SBATCH --job-name=EAS0.05
#SBATCH --error=%x.%j.err
#SBATCH --output=%x.%j.out
#SBATCH --mail-user=hzhao@teco.edu

python3 experiment_EA_shared.py --DATASET 04 --SEED 08 --e_train 0.05 --projectname VariationAware_EA_shared &
python3 experiment_EA_shared.py --DATASET 04 --SEED 09 --e_train 0.05 --projectname VariationAware_EA_shared &
python3 experiment_EA_shared.py --DATASET 05 --SEED 00 --e_train 0.05 --projectname VariationAware_EA_shared &
python3 experiment_EA_shared.py --DATASET 05 --SEED 01 --e_train 0.05 --projectname VariationAware_EA_shared &
python3 experiment_EA_shared.py --DATASET 05 --SEED 02 --e_train 0.05 --projectname VariationAware_EA_shared &
python3 experiment_EA_shared.py --DATASET 05 --SEED 03 --e_train 0.05 --projectname VariationAware_EA_shared &
python3 experiment_EA_shared.py --DATASET 05 --SEED 04 --e_train 0.05 --projectname VariationAware_EA_shared &
python3 experiment_EA_shared.py --DATASET 05 --SEED 05 --e_train 0.05 --projectname VariationAware_EA_shared &
python3 experiment_EA_shared.py --DATASET 05 --SEED 06 --e_train 0.05 --projectname VariationAware_EA_shared &
python3 experiment_EA_shared.py --DATASET 05 --SEED 07 --e_train 0.05 --projectname VariationAware_EA_shared &
python3 experiment_EA_shared.py --DATASET 05 --SEED 08 --e_train 0.05 --projectname VariationAware_EA_shared &
python3 experiment_EA_shared.py --DATASET 05 --SEED 09 --e_train 0.05 --projectname VariationAware_EA_shared &
python3 experiment_EA_shared.py --DATASET 06 --SEED 00 --e_train 0.05 --projectname VariationAware_EA_shared &
python3 experiment_EA_shared.py --DATASET 06 --SEED 01 --e_train 0.05 --projectname VariationAware_EA_shared &
python3 experiment_EA_shared.py --DATASET 06 --SEED 02 --e_train 0.05 --projectname VariationAware_EA_shared &
python3 experiment_EA_shared.py --DATASET 06 --SEED 03 --e_train 0.05 --projectname VariationAware_EA_shared &

wait
