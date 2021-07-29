import glob
from copy import deepcopy
import sys
from typing import List, Optional, Dict
from baselines.core.hv_evolution import compute_hypervolume_evolution

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from baselines import save_experiment, load_experiment

plt.style.use('ggplot')

import torch
import logging
from ax import Experiment, MultiObjectiveOptimizationConfig
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated
from matplotlib.cm import ScalarMappable

from collections import defaultdict

from tqdm import tqdm


class Visualizer:

    def __init__(
        self,
        experiments_by_types: Dict[str, List],
        **kwargs
    ):

        self.experiments_types = list(experiments_by_types.keys())
        self.experiments_by_type = experiments_by_types

        self.type_to_name = {
            'SHEMOA': 'SH-EMOA',
            'MOBOHB': 'MO-BOHB',
            'MDEHVI': 'MS-EHVI',
            'MOBANANAS': 'MO-BANANAS (SH)',
            'BNC': 'BULK & CUT',
            'EHVI': 'EHVI',
            'RS': 'Random Search',
            'ALLARCHS': 'True Paretofront'
        }

        arg_or_def = lambda k, d: kwargs[k] if k in kwargs else d
        
        SMALL_SIZE = arg_or_def('small_size', 8)
        MEDIUM_SIZE = arg_or_def('medium_size', 10)
        BIGGER_SIZE = arg_or_def('bigger_size', 12)

        # Default text sizes
        plt.rc('font', size=arg_or_def('font', SMALL_SIZE))
        # Fontsize of axes title
        plt.rc('axes', titlesize=arg_or_def('axes', SMALL_SIZE))
        # Fontsize of x and y labels
        plt.rc('axes', labelsize=arg_or_def('axes', MEDIUM_SIZE))
        # Fontsize of tick labels
        plt.rc('xtick', labelsize=arg_or_def('xtick', SMALL_SIZE))
        # Fontsize of tick labels
        plt.rc('ytick', labelsize=arg_or_def('ytick', SMALL_SIZE))
        # Legend fontsize
        plt.rc('legend', fontsize=arg_or_def('legend', SMALL_SIZE))
        # Fontsize of figure title
        plt.rc('figure', titlesize=arg_or_def('figure', BIGGER_SIZE))


    def plot_hypervolume(self, time, metric_x, metrics_y, metric_y_idx):
        
        fig, ax = plt.subplots(figsize=(10, 5))

        for exp_type in tqdm(self.experiments_types, desc='HV Overtime'):

            data = None
            exp_names = self.experiments_by_type[exp_type]
            for exp_name in exp_names:
                exp = load_experiment(exp_name)
                try:
                    hv_evolution = deepcopy(exp.hv_evolution)
                except AttributeError as e:

                    logging.warning(
                        f'Experiment {exp_name} does not have \'hv_overtime\','
                        'proceding to compute it'
                    )
                    exp.hv_evolution = compute_hypervolume_evolution(
                        exp, metric_x, metrics_y
                    )
                    save_experiment(exp, exp_name)
                    hv_evolution = deepcopy(exp.hv_evolution)

                if time:
                    hv_evolution['walltime_sec'] = pd.to_timedelta(
                        hv_evolution['walltime_sec'], unit='s'
                    )
                    hv_evolution = hv_evolution.resample(
                        '15T', on='walltime_sec'
                    ).max()[[metrics_y[metric_y_idx]]]
                    data = hv_evolution if data is None else pd.merge(
                        data, hv_evolution, how='outer', on='walltime_sec'
                    )
                else:
                    hv_evolution = hv_evolution.set_index('evaluations')
                    hv_evolution = hv_evolution[[metrics_y[metric_y_idx]]]
                    data = hv_evolution if data is None else pd.merge(
                        data, hv_evolution, how='outer', on='evaluations'
                    )

            data.columns = [
                c for i in range(len(exp_names)) for c in [
                    f'hv_{metrics_y[metric_y_idx]}_{i}'
                ]
            ]
            data = data.pad()

            if time:
                x = data.index.astype('timedelta64[h]').values
                x = np.array(x.tolist() + [24])
            else:
                x = data.index.values
                x = np.array(x.tolist() + [x[-1]])
            
            mean = data.mean(axis=1).values
            mean = np.array(mean.tolist() + [mean[-1]])
            mean[0] = 0.0

            std = data.std(axis=1).values
            std = np.array(std.tolist() + [std[-1]])
            std = std / np.sqrt(len(exp_names))

            if time:
                ax.set_xlim(0, 12)
                ax.set_xticks(np.arange(0, 13, 2))

            title = self.type_to_name[exp_type]
            ax.plot(x, mean, '-', lw=3.0, label=title)

            ax.fill_between(
                x,
                mean + std,
                mean - std,
                alpha=0.4
            )


        if time:
            ax.set_xlabel("Walltime (hours)")
            ax.set_ylim(bottom=172, top=176)
        else:
            ax.set_xlabel("Num. Evaluations")

        ax.set_ylabel("Hypervolume")

        ax.legend()
        fig.savefig(f'results/hv_overtime_{metrics_y[metric_y_idx]}.pdf')
        fig.savefig(f'results/hv_overtime_{metrics_y[metric_y_idx]}.jpg')
        plt.close(fig)

    def plot_scatter(
        self, 
        metric_x,
        metric_y,
        fraction: float=0.30,
        min_x: float=2.0,
        max_x: float=8.0,
        min_y: float=0.0,
        max_y: float=110.0
    ):
        
        num_exps = len(self.experiments_by_type)
        fig, axes = plt.subplots(
            1, num_exps, figsize=(6 * num_exps, 5)
        )

        if not isinstance(metric_x, tuple):
            metric_x = (metric_x, metric_x)
        if not isinstance(metric_y, tuple):
            metric_y = (metric_y, metric_y)

        axes[0].set_ylabel(metric_y[1])

        for i, exp_type in tqdm(enumerate(self.experiments_types), desc='Scatter'):

            max_its = 0
            total_y, total_x, total_its = [], [], []
            for exp_name in self.experiments_by_type[exp_type]:
                exp = load_experiment(exp_name)

                last_idx = [t.index for t in exp.trials.values() if (t.time_completed - exp.trials[0].time_created).total_seconds() < 12 * 3600][-1]

                data = exp.fetch_data().df
                y = data[
                    (data['metric_name'] == metric_y[0]) & (data['trial_index'] < last_idx)
                ]['mean'].values
        
                x = data[
                    (data['metric_name'] == metric_x[0]) & (data['trial_index'] < last_idx)
                ]['mean'].values
                its = np.arange(len(x))

                idx = np.random.permutation(len(x))[:int(fraction * len(x))]
                total_y += y[idx].tolist()
                total_x += x[idx].tolist()
                total_its += its[idx].tolist()

                max_its = max(max_its, its.max())

            total_its = np.array(total_its)
            idx = total_its.argsort()
            total_y = np.array(total_y)[idx]
            total_x = np.array(total_x)[idx]
            total_its = total_its[idx]

            axes[i].scatter(
                total_x, -total_y, c=total_its, alpha=0.7, vmin=0, vmax=max_its
            )
         
            axes[i].set_xlim(left=min_x, right=max_x)
            axes[i].set_ylim(top=min_y, bottom=max_y)

            axes[i].set_xlabel(metric_x[1])

            axes[i].set_title(self.type_to_name[exp_type])


            norm = plt.Normalize(0, total_its.max())
            sm = ScalarMappable(norm=norm, cmap=plt.cm.get_cmap('viridis'))

            d_tst = make_axes_locatable(axes[i])
            c_tst = d_tst.append_axes('right', size='5%', pad=0.15)
            fig.colorbar(sm, cax=c_tst, label='Iterations')

        fig.tight_layout()
        fig.savefig(f'results/scatter_{int(100 * fraction)}.pdf')
        fig.savefig(f'results/scatter_{int(100 * fraction)}.jpg')
        plt.close(fig)


    def plot_pareto_fronts(self, metric_x, metric_y):

        num_plots = min([len(v) for v in self.experiments_by_type.values()])

        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

        if not isinstance(metric_x, tuple):
            metric_x = (metric_x, metric_x)
        if not isinstance(metric_y, tuple):
            metric_y = (metric_y, metric_y)

        for i in tqdm(range(num_plots), desc='Pareto fronts'):
            
            fig, ax = plt.subplots()

            for exp_type in self.experiments_types:
                
                exp_name = self.experiments_by_type[exp_type][i]
                exp = load_experiment(exp_name)

                metrics = [
                    exp.metrics[metric_y[0]],
                    exp.metrics[metric_x[0]],
                ]

                data = exp.fetch_data().df

                last_idx = [t.index for t in exp.trials.values() if (t.time_completed - exp.trials[0].time_created).total_seconds() < 12 * 3600][-1]

                if exp_name == 'ALLARCHS':
                    last_idx = 10000000
                    print('YUJUUUUU', file=sys.stderr, flush=True)

                data = torch.tensor(np.asarray([
                    (-1 if m.lower_is_better else 1) * data[
                        (data['metric_name'] == m.name) & (data['trial_index'] < last_idx)
                    ]['mean'].values 
                    for m in metrics
                ]).T, device=device).float()

                pareto = data[is_non_dominated(data[:, [0, 1]])][:, [0, 1]]
                pareto = pareto.cpu().numpy() * [-1.0, -1.0]
                pareto = pareto[pareto[:, 0].argsort()]


                ax.plot(
                    pareto[:, 1],
                    pareto[:, 0],
                    '-o', lw=1.5, 
                    label=f'{self.type_to_name[exp_type]}'
                )

            ax.set_xlabel(metric_x[1])
            
            ax.set_ylabel(metric_y[1])

            ax.set_ylim(top=-80, bottom=-95)
            ax.set_xlim(left=0, right=1.5)

            ax.set_yticks(-np.arange(80, 95, 2))
            ticks = ax.get_yticks()
            ax.set_yticklabels([str(abs(i)) for i in ticks])

            # ticks = ax.get_xticks()
            # ax.set_xticklabels([f'$10^{int(i)}$' for i in ticks])
        
            ax.legend()

            fig.savefig(f'results/paretofronts_{i}.pdf')
            fig.savefig(f'results/paretofronts_{i}.jpg')
            plt.close(fig)


    def plot_aggregated_pareto_fronts(self, metric_x, metric_y):
        
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

        if not isinstance(metric_x, tuple):
            metric_x = (metric_x, metric_x)
        if not isinstance(metric_y, tuple):
            metric_y = (metric_y, metric_y)

        fig, ax = plt.subplots()

        sample_exp = load_experiment(
            next(v[0] for v in self.experiments_by_type.values())
        )

        for exp_type in tqdm(self.experiments_types, 'Paretofront Aggregated'):

            y, x = [], []
            for exp_name in self.experiments_by_type[exp_type]:
                exp = load_experiment(exp_name)

                metrics = [
                    exp.metrics[metric_y[0]],
                    exp.metrics[metric_x[0]],
                ]

                last_idx = [t.index for t in exp.trials.values() if (t.time_completed - exp.trials[0].time_created).total_seconds() < 12 * 3600][-1]

                if exp_name == 'ALLARCHS':
                    last_idx = 10000000
                    print('YUJUUUUU', file=sys.stderr, flush=True)


                data = exp.fetch_data().df
                data = torch.tensor(np.asarray([
                    (-1 if m.lower_is_better else 1) * data[
                        (data['metric_name'] == m.name) & (data['trial_index'] < last_idx)
                    ]['mean'].values 
                    for m in metrics
                ]).T, device=device).float()

                pareto_tst = data[is_non_dominated(data)]
                y += pareto_tst.cpu().numpy()[:, 0].tolist()
                x += pareto_tst.cpu().numpy()[:, 1].tolist()
    

            data = torch.tensor(
                np.stack([y, x], axis=1),
                device=device
            )

            del y, x

            pareto = data[is_non_dominated(data)]
            
            pareto = pareto.cpu().numpy() * [-1.0, -1.0]
            pareto = pareto[pareto[:, 0].argsort()]

            ax.plot(
                pareto[:, 1],
                pareto[:, 0],
                '-o', lw=1.5, 
                label=f'{self.type_to_name[exp_type]}'
            )

            del data
            del pareto

        ax.set_xlabel('Parameters (MB)')
        
        ax.set_ylabel('Accuracy')

        # ax.set_xlim(left=2, right=8)

        # ax.set_yticks(-np.arange(0, 110, 10))
        # ticks = ax.get_yticks()
        # ax.set_yticklabels([str(abs(i)) for i in ticks])


        # ticks = ax.get_xticks()
        # ax.set_xticklabels([f'$10^{int(i)}$' for i in ticks])

        ax.set_ylim(top=-80, bottom=-95)
        ax.set_xlim(left=0, right=1.5)

        ax.set_yticks(-np.arange(80, 95, 2))
        ticks = ax.get_yticks()
        ax.set_yticklabels([str(abs(i)) for i in ticks])

        
        ax.legend()

        fig.savefig(f'results/paretofronts_aggregated.pdf')
