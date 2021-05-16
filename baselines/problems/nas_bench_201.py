
import torch
import numpy as np
import time
#from nas_201_api import NASBench201API as API

from ax import Metric
from ax.core.search_space import SearchSpace
from ax.core.objective import MultiObjective
from ax.core.parameter import ChoiceParameter, FixedParameter, ParameterType, RangeParameter
from ax.core.outcome_constraint import ObjectiveThreshold
from ax.core.optimization_config import MultiObjectiveOptimizationConfig
from .nas_bench_search_space import NASSearchSpace
from baselines import MultiObjectiveSimpleExperiment

def get_nasbench201(name=None):

    val_acc = Metric('val_acc', True)
    tst_acc_200 = Metric('tst_acc_200', True)
    val_acc_200 = Metric('val_acc_200', True)
    num_params = Metric('num_params', True)

    objective = MultiObjective([val_acc, num_params])
    thresholds = [
        ObjectiveThreshold(val_acc, 0.0),
        ObjectiveThreshold(num_params, 2.0)
    ]
    optimization_config = MultiObjectiveOptimizationConfig(
        objective=objective,
        objective_thresholds=thresholds
    )

    params = [
        ChoiceParameter(
            name=f'p{p}', parameter_type=ParameterType.INT, values=[0, 1, 2, 3, 4]
        ) for p in range(1, 6+1)
    ]
    params.append(FixedParameter('budget', ParameterType.INT, 12))
    search_space = SearchSpace(
        parameters=params,
    )

    nasbench201 = NasBench201NPY()

    return MultiObjectiveSimpleExperiment(
        name=name,
        search_space=search_space,
        eval_function=nasbench201,
        optimization_config=optimization_config,
        extra_metrics=[tst_acc_200, val_acc_200]
    )


def get_nasbench201_cs(name=None):

    val_acc = Metric('val_acc', True)
    tst_acc_200 = Metric('tst_acc_200', True)
    val_acc_200 = Metric('val_acc_200', True)
    num_params = Metric('num_params', True)

    objective = MultiObjective([val_acc, num_params])
    thresholds = [
        ObjectiveThreshold(val_acc, 0.0),
        ObjectiveThreshold(num_params, 2.0)
    ]
    optimization_config = MultiObjectiveOptimizationConfig(
        objective=objective,
        objective_thresholds=thresholds
    )



    nasbench201 = NasBench201NPY()

    return MultiObjectiveSimpleExperiment(
        name=name,
        search_space=NASSearchSpace().as_ax_space(),
        eval_function=nasbench201,
        optimization_config=optimization_config,
        extra_metrics=[tst_acc_200, val_acc_200]
    )


# class NasBench201:
#     def __init__(self):
#         self.api = API('NAS-Bench-201-v1_1-096897.pth', verbose=False)

#     def __call__(self, x):

#         info = self._x_to_info(x)
#         val_metrics = info.get_metrics('cifar10-valid', 'x-valid')
#         tst_metrics = info.get_metrics('cifar10-valid', 'ori-test')
#         cost_metrics = info.get_compute_costs('cifar10')
#         return {
#             'val_acc': (-1.0 * val_metrics['accuracy'], 0.0),
#             'tst_acc': (-1.0 * tst_metrics['accuracy'], 0.0),
#             'num_params': (cost_metrics['params'], 0.0),        
#         }

#     def _x_to_info(self, x):
#         ops = [
#             'none',
#             'skip_connect',
#             'nor_conv_1x1', 
#             'nor_conv_3x3',
#             'avg_pool_3x3'
#         ]

#         p1, p2, p3 = ops[x['p1']], ops[x['p2']], ops[x['p3']]
#         p4, p5, p6 = ops[x['p4']], ops[x['p5']], ops[x['p6']]
#         arch = f'|{p1}~0|+|{p2}~0|{p3}~1|+|{p4}~0|{p5}~1|{p6}~2|'
#         return self.api.query_meta_info_by_index(
#             self.api.query_index_by_arch(arch)
#         )

class NasBench201NPY:
    def __init__(self):
        self.api = np.load('nasbench201_tst.npy')

    def time(self, x):
        if 'budget' not in x:
            budget = 200
        else:
            budget = x['budget']
        
        p1, p2, p3 = int(x['p1']), int(x['p2']), int(x['p3'])
        p4, p5, p6 = int(x['p4']), int(x['p5']), int(x['p6'])
        info = self.api[p1, p2, p3, p4, p5, p6]
        return info[202] * budget        

    def __call__(self, x):
        if 'budget' not in x:
            budget = 199
        else:
            budget = x['budget']
        
        p1, p2, p3 = int(x['p1']), int(x['p2']), int(x['p3'])
        p4, p5, p6 = int(x['p4']), int(x['p5']), int(x['p6'])
        info = self.api[p1, p2, p3, p4, p5, p6]

        return {
            'val_acc': (-1.0 * info[budget-1], 0.0),
            'tst_acc_200': (-1.0 * info[200], 0.0),
            'val_acc_200': (-1.0 * info[200-1], 0.0),
            'num_params': (info[201], 0.0),        
        }

    def discrete_call(self, x):
        return self(x)['num_params'][0]

