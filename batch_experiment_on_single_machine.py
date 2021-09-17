# Usage example
# python batch_experiment_on_single_machine.py --datalist testing --algorithm HANN
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
    print("Running", algorithm, "on", dataset)

    run_script = "python run_alg_on_dataset.py --algorithm ALGORITHM --dataset DATASET > results/ALGORITHM/DATASET.out"
    run_script = run_script.replace("DATASET", dataset).replace("ALGORITHM", algorithm)

    os.system(run_script)