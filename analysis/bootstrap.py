import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
import torch

from torchmetrics import MeanSquaredError, R2Score, SpearmanCorrCoef, PearsonCorrCoef
from thermompnn_benchmarking import get_metrics


def get_metrics_cpu():
    return {
        "r2": R2Score(),
        "mse": MeanSquaredError(squared=True),
        "rmse": MeanSquaredError(squared=False),
        "spearman": SpearmanCorrCoef(),
        "pearson":  PearsonCorrCoef(),
    }


def main(args):
    """Script to run bootstrap sampling on dataframe of ddG predictions for metric estimation"""
    df = pd.read_csv(args.i)
    df = df.dropna(axis=0, how='any', subset=['ddG_pred', 'ddG_true'])
    args.it = int(args.it)

    if args.gpu == 'y':
        metrics = get_metrics()
    else:
        metrics = get_metrics_cpu()

    metric_results = np.zeros((args.it, len(metrics)), dtype=np.float32)

    for i in tqdm(range(args.it)):
        sample = df.sample(frac=1, replace=True, axis=0)

        target = torch.Tensor(sample.ddG_true.values)
        preds = torch.Tensor(sample.ddG_pred.values)

        if args.gpu == 'y':
            target = target.to('cuda')
            preds = preds.to('cuda')

        for m, metric in enumerate(metrics.values()):
            metric.reset()
            result = metric(preds, target)
            metric_results[i, m] = result

    new_df = pd.DataFrame(columns=metrics.keys(), data=metric_results)

    if args.o:
        new_df.to_csv(args.o, index=True)
    return


parser = argparse.ArgumentParser()
parser.add_argument('-i', type=str, help='csv input filename')
parser.add_argument('-o', type=str, help='csv output filename')
parser.add_argument("-it", type=int, help='iterations to run bootstrapping')
parser.add_argument('-gpu', type=str, default='y', help='use gpu or no?')

args = parser.parse_args()
main(args)
