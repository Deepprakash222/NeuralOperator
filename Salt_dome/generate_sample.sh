#!/bin/bash
#SBATCH --job-name=Gempy_Dino

### File / path where STDOUT will be written, the %J is the job id

#SBATCH --output=../Results_%J.txt

### Request the time you need for execution. The full format is D-HH:MM:SS
### You must at least specify minutes or days and hours and may add or
### leave out any other parameters

#SBATCH --time=05:00:00

### Request all CPUs on one node
#SBATCH --nodes=1

### Request number of CPUs
#SBATCH --ntasks=32

#SBATCH --cpus-per-task=1


### Specify your mail address
###SBATCH --mail-user=deep.prakash.ravi@rwth-aachen.de
### Send a mail when job is done
###SBATCH --mail-type=END

### Request memory you need for your job in MB
#SBATCH --mem-per-cpu=4096M


source /home/jt925938/.bashrc
conda activate gempy_dino

python save_gempy_data.py