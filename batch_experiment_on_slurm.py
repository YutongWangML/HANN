import os 
import argparse
import UCI_classification.datasets as datasets

parser = argparse.ArgumentParser()

parser.add_argument('--datalist',
                    type = str,
                    help = 'Datalist directory name, e.g., "all" or "testing". See the "UCI_classification/data_lists/" directory.')

parser.add_argument('--algorithm',
                    type = str,
                    help = 'Algorithm name, e.g., "HANN". See "algorithms/" directory.')

FLAGS = parser.parse_args()

datalist_name = FLAGS.datalist
algorithm = FLAGS.algorithm

# check if the directory '"results/" + algorithm_name' exist
result_dir = "results/"+algorithm
if not os.path.isdir(result_dir):
    os.mkdir(result_dir)
    
datalist = datasets.get_data_list(datalist_name)

for dataset in datalist:
    jobname = algorithm+"_"+dataset+"_"
    SLURM_script = """#!/bin/bash
# The interpreter used to execute the script
#SBATCH --job-name=JOBNAME
#SBATCH --mail-user=yutongw@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=1000m 
#SBATCH --time=01-00:00:00
#SBATCH --account=clayscot0
#SBATCH --partition=standard
#SBATCH --output=/home/yutongw/HANN/results/ALGORITHM/DATASET.out
# The application(s) to execute along with its input arguments and options:
module load tensorflow/2.5.0
cd /home/yutongw/HANN/
python run_alg_on_dataset.py --algorithm ALGORITHM --dataset DATASET
    """
    SLURM_script = SLURM_script.replace("JOBNAME", jobname)\
    .replace("DATASET", dataset)\
    .replace("ALGORITHM", algorithm)

    f = open("job.slurm", "w")
    f.write(SLURM_script)
    f.close()


    cmd = 'sbatch job.slurm'  
    os.system(cmd)
    