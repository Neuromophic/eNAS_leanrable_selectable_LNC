#!/bin/bash

#SBATCH --partition=multiple
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=40
#SBATCH --mem-per-cpu=1125mb
#SBATCH --time=72:00:00
#SBATCH --job-name=pNN0.05
#SBATCH --error=%x.%j.err
#SBATCH --output=%x.%j.out
#SBATCH --mail-user=hzhao@teco.edu

python3 experiment.py --DATASET 00 --SEED 00 --e_train 0.05 --projectname VariationAwareBaseline &
python3 experiment.py --DATASET 00 --SEED 01 --e_train 0.05 --projectname VariationAwareBaseline &
python3 experiment.py --DATASET 00 --SEED 02 --e_train 0.05 --projectname VariationAwareBaseline &
python3 experiment.py --DATASET 00 --SEED 03 --e_train 0.05 --projectname VariationAwareBaseline &
python3 experiment.py --DATASET 00 --SEED 04 --e_train 0.05 --projectname VariationAwareBaseline &
python3 experiment.py --DATASET 00 --SEED 05 --e_train 0.05 --projectname VariationAwareBaseline &
python3 experiment.py --DATASET 00 --SEED 06 --e_train 0.05 --projectname VariationAwareBaseline &
python3 experiment.py --DATASET 00 --SEED 07 --e_train 0.05 --projectname VariationAwareBaseline &
python3 experiment.py --DATASET 00 --SEED 08 --e_train 0.05 --projectname VariationAwareBaseline &
python3 experiment.py --DATASET 00 --SEED 09 --e_train 0.05 --projectname VariationAwareBaseline &
python3 experiment.py --DATASET 01 --SEED 00 --e_train 0.05 --projectname VariationAwareBaseline &
python3 experiment.py --DATASET 01 --SEED 01 --e_train 0.05 --projectname VariationAwareBaseline &
python3 experiment.py --DATASET 01 --SEED 02 --e_train 0.05 --projectname VariationAwareBaseline &
python3 experiment.py --DATASET 01 --SEED 03 --e_train 0.05 --projectname VariationAwareBaseline &
python3 experiment.py --DATASET 01 --SEED 04 --e_train 0.05 --projectname VariationAwareBaseline &
python3 experiment.py --DATASET 01 --SEED 05 --e_train 0.05 --projectname VariationAwareBaseline &
python3 experiment.py --DATASET 01 --SEED 06 --e_train 0.05 --projectname VariationAwareBaseline &
python3 experiment.py --DATASET 01 --SEED 07 --e_train 0.05 --projectname VariationAwareBaseline &
python3 experiment.py --DATASET 01 --SEED 08 --e_train 0.05 --projectname VariationAwareBaseline &
python3 experiment.py --DATASET 01 --SEED 09 --e_train 0.05 --projectname VariationAwareBaseline &
python3 experiment.py --DATASET 02 --SEED 00 --e_train 0.05 --projectname VariationAwareBaseline &
python3 experiment.py --DATASET 02 --SEED 01 --e_train 0.05 --projectname VariationAwareBaseline &
python3 experiment.py --DATASET 02 --SEED 02 --e_train 0.05 --projectname VariationAwareBaseline &
python3 experiment.py --DATASET 02 --SEED 03 --e_train 0.05 --projectname VariationAwareBaseline &
python3 experiment.py --DATASET 02 --SEED 04 --e_train 0.05 --projectname VariationAwareBaseline &
python3 experiment.py --DATASET 02 --SEED 05 --e_train 0.05 --projectname VariationAwareBaseline &
python3 experiment.py --DATASET 02 --SEED 06 --e_train 0.05 --projectname VariationAwareBaseline &
python3 experiment.py --DATASET 02 --SEED 07 --e_train 0.05 --projectname VariationAwareBaseline &
python3 experiment.py --DATASET 02 --SEED 08 --e_train 0.05 --projectname VariationAwareBaseline &
python3 experiment.py --DATASET 02 --SEED 09 --e_train 0.05 --projectname VariationAwareBaseline &
python3 experiment.py --DATASET 03 --SEED 00 --e_train 0.05 --projectname VariationAwareBaseline &
python3 experiment.py --DATASET 03 --SEED 01 --e_train 0.05 --projectname VariationAwareBaseline &
python3 experiment.py --DATASET 03 --SEED 02 --e_train 0.05 --projectname VariationAwareBaseline &
python3 experiment.py --DATASET 03 --SEED 03 --e_train 0.05 --projectname VariationAwareBaseline &
python3 experiment.py --DATASET 03 --SEED 04 --e_train 0.05 --projectname VariationAwareBaseline &
python3 experiment.py --DATASET 03 --SEED 05 --e_train 0.05 --projectname VariationAwareBaseline &
python3 experiment.py --DATASET 03 --SEED 06 --e_train 0.05 --projectname VariationAwareBaseline &
python3 experiment.py --DATASET 03 --SEED 07 --e_train 0.05 --projectname VariationAwareBaseline &
python3 experiment.py --DATASET 03 --SEED 08 --e_train 0.05 --projectname VariationAwareBaseline &
python3 experiment.py --DATASET 03 --SEED 09 --e_train 0.05 --projectname VariationAwareBaseline &
python3 experiment.py --DATASET 04 --SEED 00 --e_train 0.05 --projectname VariationAwareBaseline &
python3 experiment.py --DATASET 04 --SEED 01 --e_train 0.05 --projectname VariationAwareBaseline &
python3 experiment.py --DATASET 04 --SEED 02 --e_train 0.05 --projectname VariationAwareBaseline &
python3 experiment.py --DATASET 04 --SEED 03 --e_train 0.05 --projectname VariationAwareBaseline &
python3 experiment.py --DATASET 04 --SEED 04 --e_train 0.05 --projectname VariationAwareBaseline &
python3 experiment.py --DATASET 04 --SEED 05 --e_train 0.05 --projectname VariationAwareBaseline &
python3 experiment.py --DATASET 04 --SEED 06 --e_train 0.05 --projectname VariationAwareBaseline &
python3 experiment.py --DATASET 04 --SEED 07 --e_train 0.05 --projectname VariationAwareBaseline &
python3 experiment.py --DATASET 04 --SEED 08 --e_train 0.05 --projectname VariationAwareBaseline &
python3 experiment.py --DATASET 04 --SEED 09 --e_train 0.05 --projectname VariationAwareBaseline &
python3 experiment.py --DATASET 05 --SEED 00 --e_train 0.05 --projectname VariationAwareBaseline &
python3 experiment.py --DATASET 05 --SEED 01 --e_train 0.05 --projectname VariationAwareBaseline &
python3 experiment.py --DATASET 05 --SEED 02 --e_train 0.05 --projectname VariationAwareBaseline &
python3 experiment.py --DATASET 05 --SEED 03 --e_train 0.05 --projectname VariationAwareBaseline &
python3 experiment.py --DATASET 05 --SEED 04 --e_train 0.05 --projectname VariationAwareBaseline &
python3 experiment.py --DATASET 05 --SEED 05 --e_train 0.05 --projectname VariationAwareBaseline &
python3 experiment.py --DATASET 05 --SEED 06 --e_train 0.05 --projectname VariationAwareBaseline &
python3 experiment.py --DATASET 05 --SEED 07 --e_train 0.05 --projectname VariationAwareBaseline &
python3 experiment.py --DATASET 05 --SEED 08 --e_train 0.05 --projectname VariationAwareBaseline &
python3 experiment.py --DATASET 05 --SEED 09 --e_train 0.05 --projectname VariationAwareBaseline &
python3 experiment.py --DATASET 06 --SEED 00 --e_train 0.05 --projectname VariationAwareBaseline &
python3 experiment.py --DATASET 06 --SEED 01 --e_train 0.05 --projectname VariationAwareBaseline &
python3 experiment.py --DATASET 06 --SEED 02 --e_train 0.05 --projectname VariationAwareBaseline &
python3 experiment.py --DATASET 06 --SEED 03 --e_train 0.05 --projectname VariationAwareBaseline &
python3 experiment.py --DATASET 06 --SEED 04 --e_train 0.05 --projectname VariationAwareBaseline &

wait
