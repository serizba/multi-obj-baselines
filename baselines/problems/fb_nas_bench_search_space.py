#import nas_201_api
import sys

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


class FBNASSearchSpace(CS.ConfigurationSpace):

    def __init__(self):
        super(FBNASSearchSpace, self).__init__()

        # Convolution


        p1 = UniformIntegerHyperparameter("p1", 0, 8, default_value=0, log=False)
        p2 = UniformIntegerHyperparameter("p2", 0, 8, default_value=0, log=False)
        p3 = UniformIntegerHyperparameter("p3", 0, 8, default_value=0, log=False)
        p4 = UniformIntegerHyperparameter("p4", 0, 8, default_value=0, log=False)
        p5 = UniformIntegerHyperparameter("p5", 0, 8, default_value=0, log=False)
        p6 = UniformIntegerHyperparameter("p6", 0, 8, default_value=0, log=False)

        p7 = UniformIntegerHyperparameter("p7", 0, 8, default_value=0, log=False)
        p8 = UniformIntegerHyperparameter("p8", 0, 8, default_value=0, log=False)
        p9 = UniformIntegerHyperparameter("p9", 0, 8, default_value=0, log=False)
        p10 = UniformIntegerHyperparameter("p10", 0, 8, default_value=0, log=False)
        p11 = UniformIntegerHyperparameter("p11", 0, 8, default_value=0, log=False)
        p12 = UniformIntegerHyperparameter("p12", 0, 8, default_value=0, log=False)

        p13 = UniformIntegerHyperparameter("p13", 0, 8, default_value=0, log=False)
        p14 = UniformIntegerHyperparameter("p14", 0, 8, default_value=0, log=False)
        p15 = UniformIntegerHyperparameter("p15", 0, 8, default_value=0, log=False)
        p16 = UniformIntegerHyperparameter("p16", 0, 8, default_value=0, log=False)
        p17 = UniformIntegerHyperparameter("p17", 0, 8, default_value=0, log=False)
        p18 = UniformIntegerHyperparameter("p18", 0, 8, default_value=0, log=False)

        p19 = UniformIntegerHyperparameter("p19", 0, 8, default_value=0, log=False)
        p20 = UniformIntegerHyperparameter("p20", 0, 8, default_value=0, log=False)
        p21 = UniformIntegerHyperparameter("p21", 0, 8, default_value=0, log=False)


        #p1 = CategoricalHyperparameter("p1", choices=[0,1,2,3,4], default_value=0)
        #p2 = CategoricalHyperparameter("p2", choices=[0,1,2,3,4], default_value=0)
        #p3 = CategoricalHyperparameter("p3", choices=[0,1,2,3,4], default_value=0)
        #p4 = CategoricalHyperparameter("p4", choices=[0,1,2,3,4], default_value=0)
        #p5 = CategoricalHyperparameter("p5", choices=[0,1,2,3,4], default_value=0)
        #p6 = CategoricalHyperparameter("p6", choices=[0,1,2,3,4], default_value=0)


        #self.not_mutables = ['n_conv_l', 'n_fc_l']
        self.not_mutables = ['budget']
        self.add_hyperparameters([p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13,
                                  p14, p15, p16, p17, p18, p19, p20, p21])


    def as_uniform_space(self):


        p1 = self.get_hyperparameter('p1')
        p2 = self.get_hyperparameter('p2')
        p3 = self.get_hyperparameter('p3')
        p4 = self.get_hyperparameter('p4')
        p5 = self.get_hyperparameter('p5')
        p6 = self.get_hyperparameter('p6')

        p7 = self.get_hyperparameter('p7')
        p8 = self.get_hyperparameter('p8')
        p9 = self.get_hyperparameter('p9')
        p10 = self.get_hyperparameter('p10')
        p11 = self.get_hyperparameter('p11')
        p12 = self.get_hyperparameter('p12')

        p13 = self.get_hyperparameter('p13')
        p14 = self.get_hyperparameter('p14')
        p15 = self.get_hyperparameter('p15')
        p16 = self.get_hyperparameter('p16')
        p17 = self.get_hyperparameter('p17')
        p18 = self.get_hyperparameter('p18')

        p19 = self.get_hyperparameter('p19')
        p20 = self.get_hyperparameter('p20')
        p21 = self.get_hyperparameter('p21')


        cs = CS.ConfigurationSpace()

        cs.add_hyperparameters([p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13,
                                  p14, p15, p16, p17, p18, p19, p20, p21])


        return cs

    def as_ax_space(self):
        from ax import ParameterType, RangeParameter, FixedParameter, ChoiceParameter, SearchSpace


        p1 = ChoiceParameter(name='p1', parameter_type=ParameterType.INT, values=[0, 1, 2, 3, 4, 5, 6, 7, 8])
        p2 = ChoiceParameter(name='p2', parameter_type=ParameterType.INT, values=[0, 1, 2, 3, 4, 5, 6, 7, 8])
        p3 = ChoiceParameter(name='p3', parameter_type=ParameterType.INT, values=[0, 1, 2, 3, 4, 5, 6, 7, 8])
        p4 = ChoiceParameter(name='p4', parameter_type=ParameterType.INT, values=[0, 1, 2, 3, 4, 5, 6, 7, 8])
        p5 = ChoiceParameter(name='p5', parameter_type=ParameterType.INT, values=[0, 1, 2, 3, 4, 5, 6, 7, 8])
        p6 = ChoiceParameter(name='p6', parameter_type=ParameterType.INT, values=[0, 1, 2, 3, 4, 5, 6, 7, 8])


        p7 = ChoiceParameter(name='p7', parameter_type=ParameterType.INT, values=[0, 1, 2, 3, 4, 5, 6, 7, 8])
        p8 = ChoiceParameter(name='p8', parameter_type=ParameterType.INT, values=[0, 1, 2, 3, 4, 5, 6, 7, 8])
        p9 = ChoiceParameter(name='p9', parameter_type=ParameterType.INT, values=[0, 1, 2, 3, 4, 5, 6, 7, 8])
        p10 = ChoiceParameter(name='p10', parameter_type=ParameterType.INT, values=[0, 1, 2, 3, 4, 5, 6, 7, 8])
        p11 = ChoiceParameter(name='p11', parameter_type=ParameterType.INT, values=[0, 1, 2, 3, 4, 5, 6, 7, 8])
        p12 = ChoiceParameter(name='p12', parameter_type=ParameterType.INT, values=[0, 1, 2, 3, 4, 5, 6, 7, 8])


        p13 = ChoiceParameter(name='p13', parameter_type=ParameterType.INT, values=[0, 1, 2, 3, 4, 5, 6, 7, 8])
        p14 = ChoiceParameter(name='p14', parameter_type=ParameterType.INT, values=[0, 1, 2, 3, 4, 5, 6, 7, 8])
        p15 = ChoiceParameter(name='p15', parameter_type=ParameterType.INT, values=[0, 1, 2, 3, 4, 5, 6, 7, 8])
        p16 = ChoiceParameter(name='p16', parameter_type=ParameterType.INT, values=[0, 1, 2, 3, 4, 5, 6, 7, 8])
        p17 = ChoiceParameter(name='p17', parameter_type=ParameterType.INT, values=[0, 1, 2, 3, 4, 5, 6, 7, 8])
        p18 = ChoiceParameter(name='p18', parameter_type=ParameterType.INT, values=[0, 1, 2, 3, 4, 5, 6, 7, 8])

        p19 = ChoiceParameter(name='p19', parameter_type=ParameterType.INT, values=[0, 1, 2, 3, 4, 5, 6, 7, 8])
        p20 = ChoiceParameter(name='p20', parameter_type=ParameterType.INT, values=[0, 1, 2, 3, 4, 5, 6, 7, 8])
        p21 = ChoiceParameter(name='p21', parameter_type=ParameterType.INT, values=[0, 1, 2, 3, 4, 5, 6, 7, 8])


        b = FixedParameter('budget', ParameterType.INT, 25)

       # b = FixedParameter('budget', ParameterType.INT, 25)

       # i = FixedParameter('id', ParameterType.STRING, 'dummy')

        return SearchSpace(
            parameters=[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13,
                                  p14, p15, p16, p17, p18, p19, p20, p21, b],
        )


    def sample_hyperparameter(self, hp):
        if not self.is_mutable_hyperparameter(hp):
            raise Exception("Hyperparameter {} is not mutable and must be fixed".format(hp))
        return self.get_hyperparameter(hp).sample(self.random)

    def is_mutable_hyperparameter(self, hp):
        return hp not in self.not_mutables


