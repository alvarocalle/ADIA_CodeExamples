from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from pyspark import StorageLevel

import utilities as utils
import environment as env

from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline


spark = SparkSession.builder \
    .appName("campofrio") \
    .master("local[*]") \
    .config("spark.driver.memory", "8g") \
    .config("spark.cleaner.referenceTracking.cleanCheckpoints","true") \
    .config("spark.checkpoint.compress","true") \
    .config("spark.io.compression.codec","lz4") \
    .config("spark.shuffle.mapStatus.compression.codec","lz4") \
    .config("spark.shuffle.spill.compress","true") \
    .config("spark.shuffle.compress","true") \
    .config("spark.rdd.compress","true") \
    .config("spark.driver.memory","16G")\
    .config("spark.driver.maxResultSize", "0") \
    .config("spark.kryoserializer.buffer.max", "2000M")\
    .config("spark.jars", "jars/spark-nlp-assembly-3.4.4.jar")\
    .getOrCreate()


if __name__ == '__main__':

    # lee una muestra del dataset aplanado
    data = spark.read.parquet(env.options['inputDataFile'])\
                .repartition(2*spark.getActiveSession().sparkContext.defaultParallelism, "row_id")\
                .persist(StorageLevel.MEMORY_AND_DISK)

    print('selecciona y renombra')
    data = utils.rename_columns(data, env.dict_cols_local)
    data.show(10, truncate=False)

    print('filtra obsoletos')
    data = utils.filter_obsoletos(data)
    data.show(10, truncate=False)

    print('normalizing data ...')
    data = data.withColumn(env.options['normColumn'],
                           utils.normalize_text_udf(env.options)(F.col(env.options['descColumn'])))
    data.show(10, truncate=False)

    print('brand removal ...')
    data = utils.exact_match_brand_removal(data,
                                           column_name_in=env.options['normColumn'],
                                           column_name_out=env.options['normColumnNoBrands'],
                                           list_of_brands=env.brands)
    data.show(10, truncate=False)

    # ---------------------------------------------------
    # sparkNLP pipeline for language detection
    # ---------------------------------------------------

    """
    print('sparkNLP language detection ...')
    data = utils.sparkNLP_language_detector(data,
                                            inputColumn=env.options['normColumnNoBrands'],
                                            outputColumn=env.options['langColumn'],
                                            modelpath=env.options['langDetModelFile'],
                                            threshold=env.options['langDetThreshold'])
    data.show(10, truncate=False)
    """

    # ---------------------------------------------------
    # sparkNLP pipeline for language translation
    # ---------------------------------------------------

    print('sparkNLP language translation ...')
    data = utils.sparkNLP_english_translator(data,
                                            inputColumn=env.options['normColumnNoBrands'],
                                            outputColumn=env.options['transColumn'],
                                            modelpath=env.options['langTransModelFile'])
    data.show(10, truncate=False)

    # ---------------------------------------------------
    # Find similar items
    # ---------------------------------------------------

    print('Similarity ...')
    data = utils.minhash_signatures_multilang(data,
                                              descColumn=env.options['normColumnNoBrands'],
                                              transColumn=env.options['transColumn'],
                                              ngramLength=4,
                                              numHashTables=5,
                                              threshold=0.2)
    data.show(10, truncate=False)

    # save results in parquet
    ####data.write.mode('overwrite').parquet('outputs/lang_det_results')





    # save results in parquet
    ####data.write.mode('overwrite').parquet('outputs/lang_det_results')
