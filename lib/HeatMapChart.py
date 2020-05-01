import seaborn as sns
import pandas as pd
import numpy as np

try:
    from lib.IChart import IChart
except Exception as e:
    from IChart import IChart


class HeatMapChart(IChart):
    
    def __init__(self, similarity_matrix, font_size=5, ticklabels=None, vmax=None, vmin=None):
        IChart.__init__(self)
        self.similarity_matrix = similarity_matrix
        if ticklabels:
            self.ticklabels = list(ticklabels)

        self.font_size = font_size

        if vmax is None:
            vmax = np.max(similarity_matrix)
        if vmin is None:
            vmin = np.min(similarity_matrix)
        self.vmax = vmax
        self.vmin = vmin

    def draw_chart(self):
        sns.set(font_scale=1.0)
        if isinstance(self.similarity_matrix, pd.DataFrame):
            ax = sns.heatmap(self.similarity_matrix, cmap="Oranges", 
                        xticklabels=True,
                        yticklabels=True,
                        vmax=self.vmax)
        else:
            ax = sns.heatmap(self.similarity_matrix, cmap="Oranges",
                        xticklabels=self.ticklabels, 
                        yticklabels=self.ticklabels,
                        vmax=self.vmax)

        for item in ax.get_yticklabels():
            item.set_fontsize(self.font_size)
            item.set_rotation(0)
            item.set_fontname('serif')
        for item in ax.get_xticklabels():
            item.set_fontsize(self.font_size)
            item.set_rotation(90)
            item.set_fontname('serif')