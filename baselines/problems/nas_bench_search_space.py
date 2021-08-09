#import nas_201_api
import torch
import numpy as np
#from nas_201_api import NASBench201API as API
from ax import Metric
from ax.core.search_space import SearchSpace
from ax.core.objective import MultiObjective
from ax.core.parameter import ChoiceParameter, ParameterType, RangeParameter
from ax.core.outcome_constraint import ObjectiveThreshold
from ax.core.optimization_config import MultiObjectiveOptimizationConfig
from baselines import MultiObjectiveSimpleExperiment

import ConfigSpace as CS
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
from ConfigSpace.hyperparameters import CategoricalHyperparameter


class NASSearchSpace(CS.ConfigurationSpace):

    def __init__(self):
        super(NASSearchSpace, self).__init__()

        # Convolution


        p1 = UniformIntegerHyperparameter("p1", 0, 4, default_value=0, log=False)
        p2 = UniformIntegerHyperparameter("p2", 0, 4, default_value=0, log=False)
        p3 = UniformIntegerHyperparameter("p3", 0, 4, default_value=0, log=False)
        p4 = UniformIntegerHyperparameter("p4", 0, 4, default_value=0, log=False)
        p5 = UniformIntegerHyperparameter("p5", 0, 4, default_value=0, log=False)
        p6 = UniformIntegerHyperparameter("p6", 0, 4, default_value=0, log=False)


        #p1 = CategoricalHyperparameter("p1", choices=[0,1,2,3,4], default_value=0)
        #p2 = CategoricalHyperparameter("p2", choices=[0,1,2,3,4], default_value=0)
        #p3 = CategoricalHyperparameter("p3", choices=[0,1,2,3,4], default_value=0)
        #p4 = CategoricalHyperparameter("p4", choices=[0,1,2,3,4], default_value=0)
        #p5 = CategoricalHyperparameter("p5", choices=[0,1,2,3,4], default_value=0)
        #p6 = CategoricalHyperparameter("p6", choices=[0,1,2,3,4], default_value=0)


        #self.not_mutables = ['n_conv_l', 'n_fc_l']
        self.not_mutables = ['budget']
        self.add_hyperparameters([p1, p2, p3, p4, p5, p6])


    def as_uniform_space(self):


        p1 = self.get_hyperparameter('p1')
        p2 = self.get_hyperparameter('p2')
        p3 = self.get_hyperparameter('p3')
        p4 = self.get_hyperparameter('p4')
        p5 = self.get_hyperparameter('p5')
        p6 = self.get_hyperparameter('p6')

        cs = CS.ConfigurationSpace()

        cs.add_hyperparameters([p1, p2, p3, p4, p5, p6])

        return cs

    def as_ax_space(self):
        from ax import ParameterType, RangeParameter, FixedParameter, ChoiceParameter, SearchSpace


        p1 = ChoiceParameter(name='p1', parameter_type=ParameterType.INT, values=[0, 1, 2, 3, 4])
        p2 = ChoiceParameter(name='p2', parameter_type=ParameterType.INT, values=[0, 1, 2, 3, 4])
        p3 = ChoiceParameter(name='p3', parameter_type=ParameterType.INT, values=[0, 1, 2, 3, 4])
        p4 = ChoiceParameter(name='p4', parameter_type=ParameterType.INT, values=[0, 1, 2, 3, 4])
        p5 = ChoiceParameter(name='p5', parameter_type=ParameterType.INT, values=[0, 1, 2, 3, 4])
        p6 = ChoiceParameter(name='p6', parameter_type=ParameterType.INT, values=[0, 1, 2, 3, 4])
        b = FixedParameter('budget', ParameterType.INT, 25)

       # b = FixedParameter('budget', ParameterType.INT, 25)

        # i = FixedParameter('id', ParameterType.STRING, 'dummy')

        return SearchSpace(
            parameters=[p1,p2,p3,p4,p5,p6,b],
        )


    def sample_hyperparameter(self, hp):
        if not self.is_mutable_hyperparameter(hp):
            raise Exception("Hyperparameter {} is not mutable and must be fixed".format(hp))
        return self.get_hyperparameter(hp).sample(self.random)

    def is_mutable_hyperparameter(self, hp):
        return hp not in self.not_mutables


