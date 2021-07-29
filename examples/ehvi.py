from baselines.problems.flowers import discrete_flowers
from baselines.problems import get_flowers
from baselines.problems.fashion import discrete_fashion
from baselines.problems import get_fashion
from baselines.problems import get_branin_currin, BraninCurrinEvalFunction
from baselines.problems import get_nasbench201
from baselines import save_experiment

from tqdm import tqdm
from ax import Models
from ax.modelbridge.factory import get_MOO_EHVI

if __name__ == '__main__':

    # Parameters Flowers
    # N_init = 50 # Number of initial random samples
    # N = 20000   # Number of EHVI samples (it is not important)
    # experiment = get_flowers('EHVI')  # Function to get the problem

    # Parameters Fashion
    # N_init = 10 # Number of initial random samples
    # N = 20000   # Number of EHVI samples (it is not important)
    # experiment = get_fashion('EHVI')  # Function to get the problem

    # Parameters Branin Crunin
    # N_init = 10
    # N = 25
    # experiment = get_branin_currin('EHVI')

    # Parameters NasBench201
    N_init = 10
    N = 90
    experiment = get_nasbench201('EHVI')

    #################
    #### MS-EHVI ####
    #################

    # Random search initialization
    for _ in tqdm(range(N_init), desc='Random Initialization'):
        experiment.new_trial(Models.SOBOL(experiment.search_space).gen(1))
        experiment.fetch_data()

    # Proper guided search
    for _ in tqdm(range(N), desc='EHVI'):
        
        try:
            experiment.new_trial(get_MOO_EHVI(experiment, experiment.fetch_data()).gen(1))
            experiment.fetch_data()
        except RuntimeError:
            import sys
            print('Runtime error: ', file=sys.stderr, flush=True)
            experiment.new_trial(Models.SOBOL(experiment.search_space).gen(1))
            experiment.fetch_data()
     
    save_experiment(experiment, f'{experiment.name}_100.pickle')
