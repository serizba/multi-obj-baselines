from datetime import datetime
from baselines.problems.flowers import discrete_flowers
from baselines.problems import get_flowers
from baselines.problems.fashion import discrete_fashion
from baselines.problems import get_fashion
from baselines.problems import get_branin_currin, BraninCurrinEvalFunction
from baselines.problems import get_nasbench201, NasBench201NPY
from baselines import save_experiment
from baselines.methods.msehvi.msehvi import MSEHVI

from time import time
from baselines.problems.hw_nas_bench import HW_NAS_Bench, get_hwnasbench201
from tqdm import tqdm
from ax import Models
import sys


if __name__ == '__main__':

    idx = int(sys.argv[1])

    # Parameters Flowers
    # N_init = 50 # Number of initial random samples
    # N = 20000   # Number of MS-EHVI samples (it is not important)
    # discrete_f = discrete_flowers       # Discrete function
    # discrete_m = 'num_params'           # Name of the discrete metric
    # experiment = get_flowers('MSEHVI')  # Function to get the problem

    # Parameters Fashion
    # N_init = 10 # Number of initial random samples
    # N = 20000   # Number of MS-EHVI samples (it is not important)
    # discrete_f = discrete_fashion       # Discrete function
    # discrete_m = 'num_params'           # Name of the discrete metric
    # experiment = get_fashion('MSEHVI')  # Function to get the problem

    # Parameters Branin Crunin
    # N_init = 10
    # N = 25
    # discrete_f = BraninCurrinEvalFunction().discrete_call
    # discrete_m = 'a'
    # experiment = get_branin_currin('MSEHVI')

    # # Parameters Nas-Bench-201
    # N_init = 50
    # nb201 = NasBench201NPY()
    # discrete_f = nb201.discrete_call
    # discrete_m = 'num_params'
    # experiment = get_nasbench201(f'MSEHVI_{idx}')

    # Parameters Nas-Bench-201
    N_init = 50
    nb201 = HW_NAS_Bench("nasbench201", "cifar100")
    discrete_f = nb201.discrete_call
    discrete_m = 'edgegpu_latency'
    experiment = get_hwnasbench201(f'MSEHVI_{idx}', 'cifar100')

    #################
    #### MS-EHVI ####
    #################
    curr_time = time()
    initial_time = curr_time
    # Random search initialization
    for _ in tqdm(range(N_init), desc='Random Initialization'):
        trial = experiment.new_trial(Models.SOBOL(experiment.search_space).gen(1))
        experiment.fetch_data()

        # Artificially add the time
        trial._time_created = datetime.fromtimestamp(curr_time)
        curr_time = curr_time + nb201.time(trial.arm.parameters)
        trial._time_completed = datetime.fromtimestamp(curr_time)

    # Proper guided search
    msehvi = MSEHVI(experiment, discrete_m, discrete_f)
    while curr_time - initial_time < 86400 // 2:

        try:
            msehvi.step()
        except RuntimeError as e:
            print(f'An error occurred: {e}', file=sys.stderr, flush=True)
            trial = experiment.new_trial(Models.SOBOL(experiment.search_space).gen(1))
            experiment.fetch_data()


        # Artificially add the time
        trial = list(experiment.trials.values())[-1]
        trial._time_created = datetime.fromtimestamp(curr_time)
        curr_time = curr_time + nb201.time(trial.arm.parameters)
        trial._time_completed = datetime.fromtimestamp(curr_time)
        print('Time left: ', 86400 - (curr_time - initial_time), file=sys.stderr, flush=True)
     
    save_experiment(experiment, f'{experiment.name}.pickle')
