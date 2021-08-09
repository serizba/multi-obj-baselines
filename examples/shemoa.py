import sys
from baselines.problems.hw_nas_bench import get_hwnasbench201
from baselines.problems.nas_bench_201 import get_nasbench201
from baselines.problems.simple_problems import get_branin_currin
from baselines.problems import get_flowers
from baselines.problems.flowers import FlowersSearchSpace
from baselines.problems import get_fashion
from baselines.problems.fashion import FashionSearchSpace
from baselines import save_experiment
from baselines.methods.shemoa import SHEMOA
from baselines.methods.shemoa import Mutation, Recombination, ParentSelection


if __name__ == '__main__':

    idx = sys.argv[1]

    # Parameters Flowers
    # N_init = 100
    # num_mutated = 5
    # min_budget = 5
    # max_budget = 25
    # max_function_evals = 15000
    # mutation_type = Mutation.UNIFORM
    # recombination_type = Recombination.UNIFORM
    # selection_type = ParentSelection.TOURNAMENT
    # search_space = FlowersSearchSpace()
    # non_mutable_hp = ['n_conv_l', 'n_fc_l']
    # experiment = get_flowers('SHEMOA')

    # Parameters Fashion
    # N_init = 10
    # num_mutated = 5
    # min_budget = 5
    # max_budget = 25
    # max_function_evals = 150
    # mutation_type = Mutation.UNIFORM
    # recombination_type = Recombination.UNIFORM
    # selection_type = ParentSelection.TOURNAMENT
    # search_space = FashionSearchSpace()
    # experiment = get_fashion('SHEMOA')

    # Parameters Branin Currin
    # N_init = 50
    # max_function_evals = 100
    # experiment = get_branin_currin('SHEMOA')
    # non_mutable_hp = []
    # num_mutated = 1
    # min_budget = 5
    # max_budget = 25
    # mutation_type = Mutation.UNIFORM
    # recombination_type = Recombination.UNIFORM
    # selection_type = ParentSelection.TOURNAMENT


    # # Parameters NB201
    # N_init = 50
    # max_function_evals = 1000
    # experiment = get_nasbench201(f'SHEMOA_{idx}')
    # non_mutable_hp = []
    # num_mutated = 1
    # min_budget = 5
    # max_budget = 50
    # mutation_type = Mutation.UNIFORM
    # recombination_type = Recombination.UNIFORM
    # selection_type = ParentSelection.TOURNAMENT

    # Parameters HWNB201
    N_init = 50
    max_function_evals = 1000
    experiment = get_hwnasbench201(f'SHEMOA_{idx}', 'cifar100')
    non_mutable_hp = []
    num_mutated = 1
    min_budget = 5
    max_budget = 50
    mutation_type = Mutation.UNIFORM
    recombination_type = Recombination.UNIFORM
    selection_type = ParentSelection.TOURNAMENT


    #################
    #### SH-EMOA ####
    #################
    ea = SHEMOA(
        experiment,
        non_mutable_hp,
        num_mutated,
        N_init, min_budget, max_budget,
        mutation_type=mutation_type,
        recombination_type=recombination_type,
        selection_type=selection_type,
        total_number_of_function_evaluations=max_function_evals
    )
    ea.optimize()
    save_experiment(experiment, f'{experiment.name}.pickle')
