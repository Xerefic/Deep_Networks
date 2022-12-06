#!/bin/bash

#PBS -e errorfile.err
#PBS -o logfile.log
#PBS -l select=1:ncpus=1:ngpus=1
#PBS -q gpuq

echo $PBS_O_WORKDIR
tpdir=`echo $PBS_JOBID | cut -f 1 -d .`
tempdir=$HOME/scratch/job$tpdir
mkdir -p $tempdir
cd $tempdir

cp -R $PBS_O_WORKDIR/* .

echo `which conda`
echo `conda --version`
echo `conda env list`
source ~/anaconda3/etc/profile.d/conda.sh
conda activate research-work-DAG-DNN
echo `which python`
@REM echo `conda list`
echo `locate cuda | grep /cuda$`

export PYTHONPATH="$PYTHONPATH:`pwd`/models"   
python3 -u test.py > 'checkpoints/test.txt'
python3 -u main.py > 'checkpoints/main.txt'

mv * $PBS_O_WORKDIR/.
rmdir $tempdir