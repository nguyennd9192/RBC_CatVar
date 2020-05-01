



from lib.general_lib import *
from lib.plot import plot_regression


from lib.predict import get_pred_model, multi_predict_test, multi_cv_gps
from lib.read_nlfs_out import generate_gps


def process_multi_gps(source_df_file, nlfs_out, new_instant_file,
					model_params, target_variable, kernel_type, 
					save_dir, n_components_gmm, bagsize, 
					test_points=None, train_points=None, predict_test=True,
					multi_cv_test=False):
	# this function copied from GraphicalRegression
	
	all_gps = []
	pred_models = []

	n_records = 5
	t_score = 0.80

	# , "rbf", "polynomial", "sigmoid", "laplacian"
	
	for kt in kernel_type:
		data_train = pd.read_csv(source_df_file, index_col=0)

		nlfs_out_df = pd.read_csv(nlfs_out)
		nlfs_out_df_sorted = nlfs_out_df[nlfs_out_df['best_score'] > t_score].sort_values(
									[u'best_score'], ascending=False)

		# input "new_instant_file" be either filename or dataframe obj
		if isinstance(new_instant_file, str):
			df_new_instant = pd.read_csv(new_instant_file, index_col=0)


		best_ones = nlfs_out_df_sorted.head(n_records)
		print (best_ones)

		gps = list(generate_gps(dframe=best_ones, data_buf=data_train,
								   target_variable=target_variable,
								   method="kr", kernel=kt))

		all_gps += gps
		model_params["kernel"] = kt
		# # for predict test set, including param-search for train model
		if predict_test:
			tmp_pred_model = get_pred_model(gps_extended=gps,
									df_train=data_train, df_test=df_new_instant,
									target_variable=target_variable,
									model_params=model_params)
			# model already trained
			pred_models += tmp_pred_model

	# # for predict test set
	if predict_test:

		if test_points is None:
			test_points = df_new_instant.index

		if bagsize is not None:
			for test_point in test_points:
				train_points = list(data_train.index)

				if test_point in train_points:
					train_points.remove(test_point)


				this_df_test = df_new_instant.loc[[test_point], :]
				this_df_train = data_train.loc[train_points, :]
				pred_df = multi_predict_test(pred_models=pred_models, 
					target_variable=target_variable, df_train=this_df_train, 
					df_test=this_df_test, train_idx=None, save_dir=save_dir,
					bagging=True, is_plot=True, n_components_gmm=3, # n_components_gmm
					bagsize=bagsize
					)
			# print (pred_df)
		else:
			pred_df = multi_predict_test(pred_models=pred_models, 
					target_variable=target_variable, df_train=data_train, 
					df_test=df_new_instant, train_idx=None, save_dir=save_dir,
					bagging=False, is_plot=False, n_components_gmm=3, # n_components_gmm
					bagsize=bagsize
					)
		makedirs("{0}/data_pred.csv".format(save_dir))
		pred_df.to_csv("{0}/data_pred.csv".format(save_dir))

	if multi_cv_test:
		y_obs = data_train[target_variable].values

		y_pred = multi_cv_gps(gps_extended=all_gps, y_obs=y_obs, 
			n_times_cv=1, cv_k_fold="LOO", is_fit=True)
	
		this_score = score(y_obs=y_obs, y_predict=y_pred, score_type='R2')
		this_err = error(y_obs=y_obs, y_predict=y_pred, error_type='MAE')

		plot_regression(x=y_obs, y=y_pred, tv=target_variable, dimensional="uB", n_cluster=None, 
			group_index=None, name=data_train.index,)
		title = "R2: {0}, MAE: {1}".format(round(this_score, 3), round(this_err, 3))
		plt.title(title)

		save2csv = "{0}/multi_cv.csv".format(save_dir)
		makedirs(save2csv)

		data_train["pred_cv"] = y_pred
		data_train.to_csv(save2csv)

		save2pdf = "{0}/multi_cv.pdf".format(save_dir)
		plt.savefig(save2pdf)

	return all_gps, pred_models





if __name__ == "__main__":

	config_OQMD_Co = {"source_f": "Co_des_rm_high_val", # group_0
				  "main_dir": "../input/OQMD/Co_nsp10k", # Co_nsp1k
				  "result_dir": "../result/OQMD/Co_nsp10k", # Co_nsp1k
				  "n_cluster": 5,
				  "top_k": 500
				  }
	config = config_OQMD_Co

	# source_df_file = "{0}/NLFS/{1}.csv".format(config["main_dir"], config["source_f"])
	source_df_file = "{0}/{1}.csv".format(config["main_dir"], config["source_f"])

	nlfs_out = "{0}/group_0.k8.cv10.5.laplacian.out.csv".format(config["main_dir"])
	model_params = dict({"model": "krr", 
				"is_param_search": False, # False is to take nlfs.out params as input model
				"alpha": None, "gamma": None,
				"n_folds": 10, "n_times":5}) #"LOO", 1
	kernel_type = ["laplacian"] # "rbf" "polynomial", "sigmoid", "laplacian"
	target_variable = "c_magmom_pa"



	all_gps, pred_models = process_multi_gps(source_df_file=source_df_file, 
					nlfs_out=nlfs_out,
					new_instant_file=None, 
					model_params=model_params, target_variable=target_variable,
					kernel_type=kernel_type,
					save_dir=config["result_dir"], 
					n_components_gmm=None, 
					bagsize=None,
					multi_cv_test=True, predict_test=False,)

