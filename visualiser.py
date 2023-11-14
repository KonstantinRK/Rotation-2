import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np


def plot_activation(activations, save=None):
    if len(activations.shape) == 1:
        df = pd.DataFrame(activations, columns=["activations"])
        df["activations"] = df["activations"]/df["activations"].max()
        sns.heatmap(df.transpose(), square=True, cmap="YlGnBu", cbar=False, yticklabels=False, xticklabels=False)
    elif len(activations.shape) == 3:
        def draw_p(data, **kwargs):
            data = data.pivot(index="y", columns="x", values="act")
            sns.heatmap(data, square=True, cmap="YlGnBu", cbar=False, yticklabels=False, xticklabels=False, vmin=0, vmax=1)
        data = []
        for c in range(activations.shape[-1]):
            for y in range(activations.shape[-2]):
                for x in range(activations.shape[-3]):
                    data.append([x, y, c, activations[x, y, c]])
        df = pd.DataFrame(data, columns=["x", "y", "c", "act"])
        df["act"] = df["act"] / df["act"].max()
        g = sns.FacetGrid(df, col="c")
        g.map_dataframe(draw_p)
    if save is None:
        plt.show()
    else:
        plt.savefig(save)
        plt.close()


def plot_all_activations(activations, concept_names=None, save=None):
    def draw_p(data, **kwargs):
        data = data.pivot(index="y", columns="x", values="act")
        sns.heatmap(data, square=True, cmap="YlGnBu", cbar=False, yticklabels=False, xticklabels=False, vmin=0, vmax=1)

    def draw_q(data, **kwargs):
        data = data.pivot(index="y", columns="x", values="act")
        sns.heatmap(data, square=True, cmap="YlGnBu", cbar=False, yticklabels=False, xticklabels=False, vmin=0, vmax=1)

    if len(activations.shape) == 2:
        data = []
        for con in range(activations.shape[0]):
            for x in range(activations.shape[1]):
                concept = con if concept_names is None else concept_names[con]
                data.append([concept, x, 0, activations[con, x]])
        df = pd.DataFrame(data, columns=["concept", "x", "y", "act"])
        df["act"] = df["act"] / df["act"].max()
        g = sns.FacetGrid(df, row="concept")
        g.map_dataframe(draw_p)
    elif len(activations.shape) == 4:
        data = []
        for con in range(activations.shape[-4]):
            for c in range(activations.shape[-1]):
                for y in range(activations.shape[-2]):
                    for x in range(activations.shape[-3]):
                        concept = con if concept_names is None else concept_names[con]
                        data.append([concept, x, y, c, activations[con, x, y, c]])
        df = pd.DataFrame(data, columns=["concept", "x", "y", "channel", "act"])
        df["act"] = df["act"] / df["act"].max()
        g = sns.FacetGrid(df, col="channel", row="concept")
        g.map_dataframe(draw_p)
    if save is None:
        plt.show()
    else:
        plt.savefig(save)
        plt.close()
