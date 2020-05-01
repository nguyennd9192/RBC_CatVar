from abc import ABCMeta, abstractmethod
import matplotlib.pyplot as plt
import gc

try:
    from lib.general_lib import makedirs
except Exception as e:
    from general_lib import makedirs


class IChart(object):

    def __init__(self):
        self.clr = ["red", "navy", "green", "orange", "blue", "yellow"]

        self.axis_font = {'fontname': 'serif', 'size': 14, 'labelpad': 10}
        self.title_font = {'fontname': 'serif', 'size': 14}
        self.point_size = 40
        self.colors = ["red", "navy", "green", "orange", "blue",]

    @abstractmethod
    def draw_chart(self):
        raise NotImplementedError
    
    def save_figs(self, fig, file_path):

        makedirs(file_path)

        plt.savefig(file_path, bbox_inches="tight", dpi=1000)

        fig.clf()
        plt.close()
        gc.collect()

    def set_min_max_plot(self, x, y):
        y_max = max(y)
        y_min = min(y)
        y_mean = (y_max + y_min) / 2.0
        y_std = (y_max - y_mean) / 2.0
        y_min_plot = y_mean - 2.4 * y_std
        y_max_plot = y_mean + 2.4 * y_std
    
        #plt.ylim([y_min_plot, y_max_plot])
        # plt.ylim([-10, y_max_plot])
    
    
        x_max = max(x)
        x_min = min(x)
        x_mean = (x_max + x_min) / 2.0
        x_std = (x_max - x_mean) / 2.0
        x_min_plot = x_mean - 2.4 * x_std
        x_max_plot = x_mean + 2.4 * x_std
    
        plt.xlim((x_min_plot, x_max_plot))
        plt.ylim((y_min_plot, y_max_plot))
    
        #plt.tick_params(axis='x', which='major', labelsize=20)
        #plt.tick_params(axis='y', which='major', labelsize=20)
        # plt.legend()
    
        return y_min_plot, y_max_plot