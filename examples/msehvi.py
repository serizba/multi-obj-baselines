from baselines.problems.flowers import discrete_flowers
from baselines.problems import get_flowers
from baselines.problems.fashion import discrete_fashion
from baselines.problems import get_fashion
from baselines.problems import get_branin_currin, BraninCurrinEvalFunction
from baselines.problems import get_nasbench201, NasBench201NPY
from baselines import save_experiment
from baselines.methods.msehvi.msehvi import MSEHVI

from tqdm import tqdm
from ax import Models


if __name__ == '__main__':

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

    # Parameters Nas-Bench-201
    N_init = 10
    N = 90
    discrete_f = NasBench201NPY().discrete_call
    discrete_m = 'num_params'
    experiment = get_nasbench201('MSEHVI')

    #################
    #### MS-EHVI ####
    #################

    # Random search initialization
    for _ in tqdm(range(N_init), desc='Random Initialization'):
        experiment.new_trial(Models.SOBOL(experiment.search_space).gen(1))
        experiment.fetch_data()

    # Proper guided search
    msehvi = MSEHVI(experiment, discrete_m, discrete_f)
    for _ in tqdm(range(N), desc='MSEHVI'):
        msehvi.step()
     
    save_experiment(experiment, f'{experiment.name}_100.pickle')
