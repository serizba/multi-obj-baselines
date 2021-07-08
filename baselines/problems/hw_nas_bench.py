
import torch
import numpy as np
import time
#from nas_201_api import NASBench201API as API
import sys


from ax import Metric
from ax.core.search_space import SearchSpace
from ax.core.objective import MultiObjective
from ax.core.parameter import ChoiceParameter, FixedParameter, ParameterType, RangeParameter
from ax.core.outcome_constraint import ObjectiveThreshold
from ax.core.optimization_config import MultiObjectiveOptimizationConfig
from baselines import MultiObjectiveSimpleExperiment
from .hw_nas_bench_api import HWNASBenchAPI as HWAPI
from .fb_nas_bench_search_space import FBNASSearchSpace

def get_fbnasbench(name ="None", dataset ="cifar100"):


    edgegpu_latency = Metric('edgegpu_latency', True)
    edgegpu_energy = Metric('edgegpu_energy', True)
    raspi4_latency = Metric('raspi4_latency', True)
    eyeriss_latency = Metric('eyeriss_latency', True)
    eyeriss_energy = Metric('eyeriss_energy', True)
    fpga_latency = Metric('fpga_latency', True)
    fpga_energy = Metric('fpga_energy', True)
    average_hw_metric = Metric('average_hw_metric', True)
    pixel3_latency = Metric('pixel3_latency', True)

    nasbench = HW_NAS_Bench("fbnet", dataset)
    objective = MultiObjective([edgegpu_latency, raspi4_latency])

    thresholds = [
        ObjectiveThreshold(edgegpu_latency, 100.0),
        ObjectiveThreshold(raspi4_latency, 100.0)
    ]
    optimization_config = MultiObjectiveOptimizationConfig(
        objective=objective,
        objective_thresholds=thresholds
    )

    ex = MultiObjectiveSimpleExperiment(
        name=name,
        search_space=FBNASSearchSpace().as_ax_space(),
        eval_function=nasbench,
        optimization_config=optimization_config,
        extra_metrics=[edgegpu_energy, eyeriss_latency, eyeriss_energy, fpga_latency, fpga_energy, average_hw_metric,         pixel3_latency]
    )

    print(ex)

    return ex



class HW_NAS_Bench:


    def __init__(self, search_space, dataset):
        self.search_space = search_space
        self.hw_api = HWAPI("/media/sven/Elements/hiwi5/multi-obj-baselines/HW-NAS-Bench-v1_0.pickle", search_space=self.search_space)
        self.dataset = dataset


    def __call__(self, x):

        architecture = list(x.values())

        print(architecture)
             

        HW_metrics = self.hw_api.query_by_index(architecture, self.dataset)
        value_dict = {}
        for k in HW_metrics:

            if 'average' in k:
                print("{}: {}".format(k, HW_metrics[k]))
                #continue
            elif "latency" in k:
                unit = "ms"
            else:
                unit = "mJ"

            value_dict[k] = (HW_metrics[k] , 0.0)

        return value_dict

        def discrete_call(self, x):
            return self(x)['num_params'][0]



if __name__ == "__main__":

    get_fbnasbench()











