#!/bin/bash

#SBATCH --partition=multiple
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=40
#SBATCH --mem-per-cpu=1125mb
#SBATCH --time=72:00:00
#SBATCH --job-name=batch
#SBATCH --error=%x.%j.err
#SBATCH --output=%x.%j.out
#SBATCH --mail-user=hzhao@teco.edu

script="nodetest.py"

for i in {1..80}
do
    python3 $script $i &
done
wait


#!/bin/bash

#SBATCH --partition=multiple
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=40
#SBATCH --mem-per-cpu=1125mb
#SBATCH --time=72:00:00
#SBATCH --job-name=batch
#SBATCH --error=%x.%j.err
#SBATCH --output=%x.%j.out
#SBATCH --mail-user=hzhao@teco.edu

python3 scriptname1.py 1 &
python3 scriptname2.py 2 &
python3 scriptname3.py 3 &
python3 scriptname4.py 4 &
python3 scriptname5.py 5 &

wait
