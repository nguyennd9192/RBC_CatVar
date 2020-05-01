
import pandas as pd
import numpy as np
from lib.HeatMapChart import HeatMapChart
import matplotlib.pyplot as plt




def display_large_dataframe(large_matrix, output_dir, is_sample=True, df_tick_name=None, size_submt=200):
	# if is_sample is True, then use original idx, others use df_tick_name


	n_instances = large_matrix.values.shape[0]
	list_index = np.array_split(large_matrix.index.values, int(n_instances / size_submt))
	i = 0
	vmax = np.max(large_matrix.values)
	for rows_index, rows_label in enumerate(list_index):
		for columns_index, columns_label in enumerate(list_index):
			row_inst, column_inst = large_matrix.loc[rows_label, columns_label].values.shape
			if is_sample:
				df_tmp = pd.DataFrame(
					large_matrix.loc[rows_label, columns_label].values, 
					columns=columns_label,
					index=rows_label
				)
			elif df_tick_name is not None:
				df_tmp = pd.DataFrame(
					large_matrix.loc[rows_label, columns_label].values, 
					columns=df_tick_name.loc[columns_label],
					index=df_tick_name.loc[rows_label]
				)
			else:
				print ("None value of df_tick_name args")
			heat_map_figure = plt.figure(figsize=(9, 8))
			heat_map_chart = HeatMapChart(similarity_matrix=df_tmp, vmax=vmax, font_size=3)
			heat_map_chart.draw_chart()
			heat_map_chart.save_figs(heat_map_figure, "{}/sub_hm_R{}-{}_C{}-{}.pdf".format(
					output_dir, rows_index * row_inst, (rows_index + 1) * row_inst - 1,
					columns_index * column_inst, (columns_index + 1) * column_inst - 1
				)
			)
			i += 1