import sys
import os
import traceback
import yaml
import time
import itertools
import pandas as pd
from argparse import ArgumentParser
from datetime import datetime, timedelta

try:
    from lib.read_load_config import read_config, load_config
    from lib.core_spark import spark_cal
    from lib.ini_sampling import rand_gidx_sampling

except Exception as e:
    from read_load_config import read_config, load_config
    from core_spark import spark_cal
    from ini_sampling import rand_gidx_sampling


def parse_arguments():
    parser = ArgumentParser(description="Regression-based clustering")
    parser.add_argument("--partitions", metavar="N", type=int, required=True,
                        help="The number of partitions for dividing feature combinations to nodes in Spark.")
    parser.add_argument("--sort", action="store_true",
                        help="Perform sorting the output on the Spark.")
    parser.add_argument("--estimate-speed", metavar="M", type=int,
                        help="Estimate the computation speed by running with M combinations.")
    parser.add_argument("config", metavar="FILE",
                        help="Config file.")
    return parser.parse_args()


def main():
    from pyspark import SparkContext

    args = parse_arguments()

    start_time = datetime.now()

    print("Start reading config")
    print("========================")
    cf = load_config(cfg_file=os.path.abspath(args.config))

    if cf is None:
        print("Error on loading config file.")
        return 1

    config, function = read_config(cf)

    for key, value in config.items():
        print("{0}: {1}".format(key, value))

    print("Success in reading config")
    print("========================")

    target_variable = config["target_variable"]
    remove_variable = config["remove_variable"]
    predicting_variables = config["predicting_variables"]

    data_df = pd.read_csv(config["input_file"], index_col=0)
    n_instance = len(data_df.index)

    inits = []
    if function["RBC_score_var"] == "Active":
        config["visualize"] = False

        for _ in range(config["RBC_ntimes"]):
            this_group_index = rand_gidx_sampling(
                n_cluster=config["n_cluster"], length=n_instance)
            inits.append(this_group_index)

    elements = list(
        enumerate(list(itertools.product(predicting_variables, inits))))

    # TODO: backup things like this
    """
    if os.path.exists(config["output"]["file"]):
        df = pd.read_csv(config["input"]["file"])
        csv_nfc = df.shape[0]
    #     # with open("{}.crash".format(uuid.uuid4().hex), "w") as f:
    #     #     f.write("{}".format(csv_nfc))
        fcg = fcg[csv_nfc:]

    """ 

    duration = (datetime.now() - start_time).total_seconds()
    print("Time for pre-processing data: {} (s)".format(duration))

    ##
    # Spark
    ##
    start_time = datetime.now()

    sc = SparkContext(appName="RBC - {}".format(config["config_name"]))
    n_eles = len(elements)
    print("Number of Initial Vectors: {0}".format(n_eles))

    try:
        # The data is big, thus broadcast it to node.
        data = sc.broadcast(data_df)

        groups = [elements[i:i + 2**13]
                  for i in range(0, len(elements), 2**13)]

        for group in groups:
            # Divide feature combinations to the number of partitions.
            n_eles = len(group)
            if n_eles < args.partitions:
                rdd = sc.parallelize(group, numSlices=n_eles)
            else:
                rdd = sc.parallelize(group, numSlices=args.partitions)

            rs_regression = rdd.map(
                lambda e: (e[1], spark_cal(data_df=data.value, element=e[1],
                                           target_variable=target_variable,
                                           config=config,
                                           index_job=e[0])))

            result = rs_regression.collect()

            ##
            # Write the result
            ##
            # output = pd.DataFrame.from_records(result)
            output = pd.DataFrame.from_records(map(lambda x: x[1], result))

            # print(config['out_dir'])
            output_file = os.path.join(config['out_dir'],
                                       "{}.nc{}.out.csv".format(os.path.basename(config['input_file']),
            config['n_cluster']))
            
            output.to_csv(output_file, index=False)
            # if not os.path.exists(output_file):
            #     os.makedirs(config['out_dir'])
            #     output.to_csv(output_file, index=False)
            #     print("Output file: {}".format(output_file))
            # else:
            #     output.to_csv(output_file, index=False, mode="a", header=False)
            #     print("Appended to {}".format(output_file))

            print("Output file: {}".format(output_file))

    except KeyboardInterrupt:
        print("\nProgram terminated by Ctrl +C ")
    finally:
        sc.stop()

    duration=(datetime.now() - start_time).total_seconds()
    print("Time for running on Spark: {} (s)".format(duration))


if __name__ == "__main__":
    sys.exit(main())
