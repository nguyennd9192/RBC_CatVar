

import matplotlib.offsetbox as offsetbox
from sklearn.model_selection import cross_val_predict


try:
	from lib.general_lib import *
	from lib.normalize import get_pv_Xnorm_y, get_Xnorm
	from lib.lasso_in_cluster import lasso_in_cluster
	from lib.plot import set_plot_configuration
	from lib.kr_parameter_search import get_estimator, alpha_search
	from lib.gA_pred_gB import gA_pred_gB
except Exception as e:
	from general_lib import *
	from normalize import get_pv_Xnorm_y, get_Xnorm
	from lasso_in_cluster import lasso_in_cluster
	from plot import set_plot_configuration
	from kr_parameter_search import get_estimator, alpha_search
	from gA_pred_gB import gA_pred_gB


color_gs = {
            0: 'red',
            1: 'blue',
            2: 'orange',
            3: 'green',
            4: 'brown',
            5: 'violet',
            6: 'grey',
            7: 'purple'
 }
size_point = 20
size_text = 10
alpha_point = 0.7
title_font = {'fontname': 'serif', 'size': 12}


def add_subplot(fig, row, col, nrows, ncols, x, y, color, tv=None, dimensional=None, text=None):
	figure_position = row * ncols + col + 1 

	ax = fig.add_subplot(nrows, ncols, figure_position)
	y_min_plot, y_max_plot = set_plot_configuration(x=x,y=y,
				tv=tv, dimensional=dimensional, size_fig='small')

	x_ref = np.linspace(y_min_plot, y_max_plot, 100)
	ax.plot(x_ref, x_ref, linestyle='-.', c='red', alpha=0.8)
	ax.tick_params(axis='both', labelbottom=False, labelleft=False)

	ax.scatter(x, y, s=size_point, alpha=alpha_point, c=color)
	#for this_i, this_name in enumerate(name_in_this_cluster):
	#    plt.annotate(this_name, (y[this_i], y_predicted[this_i]), size=7)
	# add legend
	ob = offsetbox.AnchoredText(text, loc=4, prop=dict(fontsize=8))
	ax.add_artist(ob)
	return fig


def confusion_matrix(df, inst_idx, group_index, pv, tv, result_dir,
				params=None):

	predict_model = params["predict_model"]
	dimensional = params["dimensional"]
	rm_v = params["remove_variable"]
	visualize = params["visualize"]
	n_cv = params["n_cv"]
	n_times = params["n_cv"]
	# get pv, Xnorm and y for all
	pv, X, y, instance_name = get_pv_Xnorm_y(df=df, inst_idx=inst_idx, tv=tv, pv=pv, rm_v=rm_v)

	# get n_cluster
	n_cluster = max(group_index) + 1

	predict_df = pd.DataFrame(index=instance_name, 
		columns=['g_{0}'.format(k) for k in range(n_cluster)])

	tmp_name = "{0}/cfs_mt_nc{1}.txt".format(result_dir, n_cluster)
	makedirs(tmp_name)
	cfs_mt_file = open(tmp_name, "w")


	group_index, alpha_bests, x_after_isomap, \
	score_all_grps, err_all_grps, score_total_weight, \
	score_gs, error_gs, y_obs_gs, y_pred_gs = lasso_in_cluster(df=df, inst_idx=inst_idx, 
		 pv=pv, tv=tv, result_dir=result_dir, n_cluster=n_cluster,
		group_index=group_index, lasso_revise=None, params=params)
	

	score_matrix = np.empty([n_cluster, n_cluster])
	if visualize:
		fig = plt.figure(figsize=(8, 8))
		title = "Overview: estimator model: {0} \n R2: {1}, MAE: {2} ({3}), R2 weight: {4}".format(
			predict_model, round(score_all_grps, 3), 
			round(err_all_grps, 3), dimensional, round(score_total_weight, 3))

		plt.title(title, **title_font)

	tmp_name = "{0}/coeff_vect_{1}.txt".format(result_dir, n_cluster)
	makedirs(tmp_name)
	file_out_coeff = open(tmp_name, "w")
	file_out_coeff.write("{0}, \"intercept\"\n".format(pv))


	for g in range(n_cluster):
		name_g = instance_name[np.where(group_index == g)]
		pv_g, X_g, y_g, name_g = get_pv_Xnorm_y(df=df, inst_idx=name_g,
				tv=tv, pv=pv, rm_v=rm_v)


		estimator = get_estimator(predict_model=predict_model)
		estimator.alpha = alpha_bests[g]
		estimator.fit(X=X_g, y=y_g)
		y_pred_g = estimator.predict(X=X_g)
		print (len(name_g), len(df.index))
		predict_df.loc[name_g, 'g_{0}'.format(g)] = y_pred_g
		
		# print out
		file_out_coeff.write("Coeff group {0} \n {1}, {2}\n".format(g, estimator.coef_, estimator.intercept_) )

		# for Pearson score
		# score_g2g = np.corrcoef(y_g, y_pred_g)[0, 1]
		score_g2g = score(y_obs=y_g, y_predict=y_pred_g, score_type='R2')
		error_g2g = error(y_obs=y_g, y_predict=y_pred_g) # score

		if visualize:
			text = "R: {0} \nMAE: {1} ({2})".format(round(score_g2g, 3), 
				round(error_g2g, 3), dimensional)
			fig = add_subplot(fig=fig, row=g, col=g, nrows=n_cluster, ncols=n_cluster, 
				x=y_g, y=y_pred_g, color=color_gs[g], tv=tv, dimensional=dimensional, text=text)

			# Dam ss add
			#if figure_position == 16:
			#    plt.figure(figsize=(8, 8))
			#    plt.scatter(y, y_predicted, s=size_point,
			#                alpha=alpha_point, c=option[k])
			#    for this_i, this_name in enumerate(name_k):
			#        plt.annotate(this_name, (y[this_i], y_predicted[this_i]), size=5)
			#    plt.savefig("{0}/test_green.pdf".format(dir))
			#    extract_df = df.loc[name_k][:]
			#    extract_df["y_predict"] = y_predicted
			#    extract_df.to_csv("{0}/test_green.csv".format(dir))

		score_matrix[g][g] = score_g2g
		cfs_mt_file.write("model: {0}, data: {1}, corr_coeff: {2}, RMSE: {3}".format(
							g, g, score_g2g, error_g2g))

		# to prepare loop for others cluster diff 2 g
		gen = (x for x in range(n_cluster) if x != g)

		for k in gen:
			name_k = instance_name[np.where(group_index == k)]

			# pred_model_g = estimator
			pred_model_g = get_estimator(predict_model=predict_model)
			pred_model_g.alpha = alpha_bests[g]

			y_k, y_pred_k, err_out = gA_pred_gB(df=df, pred_model=pred_model_g, 
									inst_gA=name_g, inst_gB=name_k,
							    	tv=tv, pv=pv, rm_v=rm_v)

			predict_df.at[name_k, 'g_{0}'.format(k)] = y_pred_k

			# for Pearson score
			score_g2k = np.corrcoef(y_k, y_pred_k)[0, 1]
			# score_g2k = score(y_obs=y_g, y_predict=y_pred_g, score_type='R2')
			error_g2k = error(y_obs=y_k, y_predict=y_pred_k)
			
			score_matrix[g][k] = score_g2k

			if visualize:
				text = "R: {0} \nMAE: {1} ({2})".format(round(score_g2k, 3), 
										round(error_g2k, 3), dimensional)

				fig = add_subplot(fig=fig, row=g, col=k, 
					nrows=n_cluster, ncols=n_cluster,
					color=color_gs[k], 
					x=y_k, y=y_pred_k, 
					tv=tv, dimensional=dimensional, text=text)

				cfs_mt_file.write("model: {0}, data: {1} \n corr_coeff: {2}, RMSE: {3}".format(g, k, score_g2k, error_g2k))
				cfs_mt_file.write("=======================\n")
	if visualize:
		save_at = "{0}/cfs_mtrix_nc{1}.pdf".format(result_dir, n_cluster)
		makedirs(save_at)
		plt.savefig(save_at)
		print ("Save at", save_at)
		release_mem(fig)

	save_at = '{0}/predict_df_{1}.csv'.format(result_dir, n_cluster)
	makedirs(save_at)
	print ("Save at", save_at)

	predict_df.to_csv(save_at)
	print ("score_matrix: ", score_matrix)

	return score_matrix, group_index


if __name__ == '__main__':

	input_dir = "/home/nguyen/Bureau/work/RBC_voting/input/OQMD/Co_nsp10k"
	result_dir = "/home/nguyen/Bureau/work/RBC_voting/result/OQMD/Co_nsp10k"


	source_file = "{0}/Co_des_rm_high_val.csv".format(input_dir)
	gidx_file = "{0}/clustering_result_5_top_20.csv".format(result_dir)
	

	source_df = pd.read_csv(source_file, index_col=0)
	gidx_df = pd.read_csv(gidx_file, index_col=0)

	pv = ["s2", "d7", "s1-s2", "s2-s1", "s2-s2", "s2-d1", "s2-d7", 
	"s2-d10", "s2-f14", "d1-s2", "d7-s2", "d7-d10", "d10-s2", 
	"d10-d7", "f14-s2"]
	tv = "c_magmom_pa"

	group_index = gidx_df["result1"].values


	save_result_dir = result_dir + "/confusion_matrix"
	confusion_matrix(df=source_df, inst_idx=source_df.index, group_index=group_index, 
		pv=pv, tv=tv, result_dir=save_result_dir,
		params=dict({"predict_model":'Lasso', "dimensional":'uB', 
			"rm_v":None, "n_cv":10, "n_times":3}))








