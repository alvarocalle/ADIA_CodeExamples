from pyspark.sql import functions as F
from pyspark.sql import types as T

from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("campofrio").config("spark.driver.memory", "8g") \
    .config("spark.cleaner.referenceTracking.cleanCheckpoints","true") \
    .config("spark.checkpoint.compress","true") \
    .config("spark.io.compression.codec","lz4") \
    .config("spark.shuffle.mapStatus.compression.codec","lz4") \
    .config("spark.shuffle.spill.compress","true") \
    .config("spark.shuffle.compress","true") \
    .config("spark.rdd.compress","true") \
    .getOrCreate()



file = '../alex/data/fabricantes.txt'
data = spark.read.text(file) \
        .withColumnRenamed('value', 'marcas') \
        .select('marcas')

data.show(10, truncate=False)
    
# save results in parquet
data.write.mode('overwrite').parquet('outputs/fabricantes')
