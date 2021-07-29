from ax.modelbridge.factory import get_MOO_EHVI
import torch
import numpy as np
import pandas as pd

from ax import Experiment, MultiObjectiveOptimizationConfig
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated

import tqdm


def compute_hypervolume_evolution(
    experiment: Experiment, 
    metric_x,
    metrics_y,
    device=None):
    assert isinstance(
        experiment.optimization_config, MultiObjectiveOptimizationConfig
    ), 'experiment must have an optimization_config of type' \
       'MultiObjectiveOptimizationConfig '

    device = device or torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    metrics = [
        *[experiment.metrics[m] for m in metrics_y],
        experiment.metrics[metric_x],
    ]

    thresholds = {
        th.metric.name: th 
        for th in experiment.optimization_config.objective_thresholds
    }
    thresholds = [
        thresholds[metrics_y[0]],
        thresholds[metric_x],
    ]

    data = experiment.fetch_data().df
    data = torch.tensor(np.asarray([
        (-1 if m.lower_is_better else 1) * data[
            data['metric_name'] == m.name
        ]['mean'].values 
        for m in metrics
    ]).T, device=device).float()
   
    hv = Hypervolume(torch.tensor([
        (-1 if th.metric.lower_is_better else 1) * th.bound 
        for th in thresholds
    ], device=device))

    hv_results = {
        'evaluations': [0.0],
        'walltime_sec': [0.0],
    }
    for m in metrics_y:
        hv_results[m] = [0.0]

    for i in tqdm.tqdm(range(1, len(data)), desc=experiment.name):
        for idx, m in enumerate(metrics_y):
            hv_results[m].append(
                hv.compute(
                    data[:i, [idx, -1]][is_non_dominated(data[:i, [idx, -1]])]
                )
            )
        hv_results['walltime_sec'].append(
            (experiment.trials[i - 1].time_completed - experiment.time_created)
            .total_seconds()
        )
        hv_results['evaluations'].append(i)

    return pd.DataFrame(hv_results)
