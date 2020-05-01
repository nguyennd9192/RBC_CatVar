. /opt/trump/vars.sh

mkdir ./tmp
find ./ -name "*.py" -exec zip ./tmp/rbc.zip {} \;

MASTER=spark://iserlohn-ib:7077
NP=224

CONFIG=$1

# PYSPARK_PYTHON=$(readlink -f $(which python)) spark-submit --py-files ./tmp/nlfs.zip --master ${MASTER} --driver-memory 16G --driver-cores 4 --num-executors 32 --executor-cores 4 --executor-memory 7G --conf spark.default.parallelism=$(( $NP * 4 )) nlfs-spark-backup.py --partitions $(( $NP * 4 )) ${CONFIG}

PYSPARK_PYTHON=$(readlink -f $(which python)) spark-submit --py-files ./tmp/rbc.zip --master ${MASTER} --driver-memory 16G --driver-cores 4 --num-executors 28 --executor-cores 8 --executor-memory 62G --conf spark.default.parallelism=$(( $NP * 8 )) main_rbc_spark.py --partitions $(( $NP * 8 )) ${CONFIG}
