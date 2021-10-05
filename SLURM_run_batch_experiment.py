import os 
import argparse
import UCI_classification.datasets as datasets

parser = argparse.ArgumentParser()

parser.add_argument('--datalist',
                    type = str,
                    help = 'Datalist directory name, e.g., "all" or "testing". See the "UCI_classification/data_lists/" directory.')

parser.add_argument('--algorithm',
                    type = str,
                    help = 'Algorithm name, e.g., "HANN15OCE". See "algorithms/" directory.')

parser.add_argument('--cpus',
                    type = int, 
                    default=1,
                    help = 'Number of cpus to use (default = 1)')


FLAGS = parser.parse_args()

datalist_name = FLAGS.datalist
algorithm = FLAGS.algorithm
cpus = FLAGS.cpus

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
#SBATCH --mail-user=YOUR_EMAIL
#SBATCH --mail-type=BEGIN,END
#SBATCH --cpus-per-task=NCPUS
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=1000m 
#SBATCH --time=01-00:00:00
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=standard
#SBATCH --output=COMPLETE_DIRECTORY_TO_HANN/results/ALGORITHM/DATASET.out
# The application(s) to execute along with its input arguments and options:
module load tensorflow/2.5.0
cd COMPLETE_DIRECTORY_TO_HANN/HANN/
python run_alg_on_dataset.py --algorithm ALGORITHM --dataset DATASET
    """
    SLURM_script = SLURM_script.replace("JOBNAME", jobname)\
    .replace("DATASET", dataset)\
    .replace("NCPUS", str(cpus))\
    .replace("ALGORITHM", algorithm)

    f = open("job.slurm", "w")
    f.write(SLURM_script)
    f.close()


    cmd = 'sbatch job.slurm'  
    os.system(cmd)
    