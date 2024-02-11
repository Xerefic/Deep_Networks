#!/bin/bash

#PBS -e errorfile.err
#PBS -o logfile.log
#PBS -l select=1:ncpus=1qsu:ngpus=1
#PBS -q gpuq

echo $PBS_O_WORKDIR
tpdir=`echo $PBS_JOBID | cut -f 1 -d .`
tempdir="$HOME/scratch/job$tpdir"
mkdir -p $tempdir
cd $tempdir

cp -R $PBS_O_WORKDIR/* .

module load anaconda3_2023
echo `which conda`
echo `conda --version`
echo `conda env list`
export PATH="/lfs/usrhome/btech/me20b032/.conda/envs/torch/bin:$PATH"
source /lfs/sware/anaconda3_2023/etc/profile.d/conda.sh
conda activate torch
echo `which python`
echo `locate cuda | grep /cuda$`

export PYTHONPATH="$PYTHONPATH:`pwd`/models"
python3 -u main.py > main.txt

mv * $PBS_O_WORKDIR/.
rmdir $tempdir