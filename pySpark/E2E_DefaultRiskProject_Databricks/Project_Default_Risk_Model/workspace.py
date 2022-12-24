# Databricks notebook source
import pandas as pd

df = spark.createDataFrame(
  pd.DataFrame({'col1':[1,2,3,4], 'col2':[1,2,3,4]})
)

# COMMAND ----------

spark.createDataFrame(pd.DataFrame({'col1':[1,2,3,4], 'col2':[1,2,3,4]})) .show()

# COMMAND ----------

#save dataframe to file_path_CSV folder using CSV, using coalesce to get only one partition
storage_account_name = 'chqd1weustahqdatacrit002'
container_name = 'lab-demodatascience'
folder_path_OUT = '/OUT'
csv_name = '/nulls.csv'
file_path_CSV = f"abfss://{container_name}@{storage_account_name}.dfs.core.windows.net{folder_path_OUT}{csv_name}"

df_write = df.coalesce(1)\
            .write.format("csv")\
            .option("header",True)\
            .option("delimiter","|").save(file_path_CSV)