from baselines.problems import get_flowers
from baselines.problems.flowers import FlowersSearchSpace
from baselines.problems import get_fashion
from baselines.problems.fashion import FashionSearchSpace
from baselines import save_experiment
from baselines.methods.mobananas import BANANAS_SH
from baselines.methods.mobananas import Neural_Predictor
from baselines.problems import NASSearchSpace
from baselines.problems import get_nasbench201_cs, NasBench201NPY
from baselines.problems import NASSearchSpace
from datetime import datetime
from time import time
from baselines.problems.hw_nas_bench import HW_NAS_Bench, get_hwnasbench201
from tqdm import tqdm
from ax import Models
import sys

if __name__ == '__main__':

    idx = int(sys.argv[1])

    # Parameters Flowers
    #N_init = 10
    #min_budget = 5
    #max_budget = 25
    #max_function_evals = 10000
    #num_arch=20
    #select_models=10
    #eta=3
    #search_space = FlowersSearchSpace()
    #experiment = get_flowers('MOSHBANANAS')

    # Parameters Fashion
    # N_init = 10
    # min_budget = 5
    # max_budget = 25
    # max_function_evals = 400
    # num_arch=20
    # select_models=10
    # eta=3
    # search_space = FashionSearchSpace()
    # experiment = get_fashion('MOSHBANANAS')

    # nb201 = NasBench201NPY()
    # experiment = get_nasbench201_cs(f'MOBANANAS_{idx}')
    # search_space = NASSearchSpace()
    # initial_samples = 20
    # min_budget = 10
    # max_budget = 200
    # function_evaluations = 10000
    # num_arch=20
    # select_models=10
    # eta=3

    nb201 = HW_NAS_Bench("nasbench201", "cifar100")
    experiment = get_hwnasbench201(f'MOBANANAS_{idx}', 'cifar100')
    search_space = NASSearchSpace()
    initial_samples = 20
    min_budget = 10
    max_budget = 200
    function_evaluations = 10000
    num_arch=20
    select_models=10
    eta=3



    #####################
    #### MOSHBANANAS ####
    #####################

    neural_predictor = Neural_Predictor(num_epochs = 80, num_ensamble_nets = 5)
    banana = BANANAS_SH(neural_predictor, experiment, search_space, initial_samples, num_arch, max_budget,min_budget, eta,  select_models, function_evaluations)
 
    curr_time = time()
    initial_time = curr_time

    while curr_time - initial_time < 86400 // 2:
        print(type(banana))
        time_budget = curr_time - initial_time
        banana.step(curr_time, initial_time, 86400 // 2, nb201)

        trial = list(experiment.trials.values())[-1]
        trial._time_created = datetime.fromtimestamp(curr_time)
        curr_time = curr_time + nb201.time(trial.arm.parameters)
        trial._time_completed = datetime.fromtimestamp(curr_time)
        print('Time left: ', 86400 - (curr_time - initial_time), file=sys.stderr, flush=True)



    save_experiment(experiment, f'{experiment.name}.pickle')
