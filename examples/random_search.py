from datetime import datetime
from ax import Models

from tqdm import tqdm
from time import time
import sys

from baselines.problems.hw_nas_bench import HW_NAS_Bench, get_hwnasbench201
from baselines.problems.nas_bench_201 import NasBench201NPY
from baselines.problems import get_flowers
from baselines.problems import get_branin_currin
from baselines.problems import get_fashion
from baselines.problems import get_nasbench201
from baselines.problems import get_fbnasbench
from baselines import save_experiment

if __name__ == '__main__':

    idx = int(sys.argv[1])

    # Parameters Flowers
    # N = 20000   # Number of samples (it is not important)
    # experiment = get_flowers('RandomSearch')  # Function to get the problem

    # Parameters Fashion
    # N = 20000   # Number of samples (it is not important)
    # experiment = get_fashion('RandomSearch')  # Function to get the problem

    # Parameters Nas-Bench-201
   #  N = 100
   #  nb201 = NasBench201NPY()
   #  experiment = get_nasbench201(f'RandomSearch_{idx}')

    # Parameters HW-Nas-Bench-201
    N = 100
    hwnb201 = HW_NAS_Bench("nasbench201", "cifar100")
    experiment = get_hwnasbench201(f'RandomSearch_{idx}', 'cifar100')

    ######################
    ### Random Search ####
    ######################

    curr_time = time()
    initial_time = curr_time
    while curr_time - initial_time < 86400 // 2:
       trial = experiment.new_trial(Models.SOBOL(experiment.search_space).gen(1))
       d = experiment.fetch_data()

       print(d.df['metric_name'].unique())

       # Artificially add the time
       trial._time_created = datetime.fromtimestamp(curr_time)
       curr_time = curr_time + hwnb201.time(trial.arm.parameters)
       trial._time_completed = datetime.fromtimestamp(curr_time)

       print('Time left: ', 86400 - (curr_time - initial_time), file=sys.stderr, flush=True)

    save_experiment(experiment, f'{experiment.name}.pickle')
