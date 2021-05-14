from ax import Models
from tqdm import tqdm

from baselines.problems import get_flowers
from baselines.problems import get_branin_currin
from baselines.problems import get_fashion
from baselines.problems import get_nasbench201
from baselines import save_experiment

if __name__ == '__main__':

    # Parameters Flowers
    # N = 20000   # Number of samples (it is not important)
    # experiment = get_flowers('RandomSearch')  # Function to get the problem

    # Parameters Fashion
    # N = 20000   # Number of samples (it is not important)
    # experiment = get_fashion('RandomSearch')  # Function to get the problem

    # Parameters Nas-Bench-201
    N = 100
    experiment = get_nasbench201('RandomSearch')

    #######################
    #### Random Search ####
    #######################
    for _ in tqdm(range(N), desc='Random Search'):
        experiment.new_trial(Models.SOBOL(experiment.search_space).gen(1))
        experiment.fetch_data()

    print(experiment.fetch_data().df)
    save_experiment(experiment, f'{experiment.name}_100.pickle')
