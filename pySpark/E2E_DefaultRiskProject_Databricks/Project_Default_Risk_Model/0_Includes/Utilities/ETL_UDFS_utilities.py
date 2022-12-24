# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC #### ETL UDFS utilities
# MAGIC 
# MAGIC - Definition of user defined functions that support the ETL process batch, streaming.

# COMMAND ----------

def read_data_from_delta(dict_config, columns_to_keep, to_pandas=False):
  """ function that retrieve data from a Delta path and returns a dataframe
    Parameters:
        dict_config (dict) : dictionary with info containing storage_account_name, container_name
                             folder_path and file_name
        columns_to_keep (list) : list of columns to keep when reading the data
        to_pandas (bool): whether we want a pandas (True) or a spark (False) dataframe
    Return:
        df : pandas or spark dataframe

    Examples:
    >>> dict_config = {
    >>>   'storage_account_name' : '****',
    >>>   'container_name' : '****',
    >>>   'folder_path' : '****',
    >>>   'file_name' : '****'
    >>> }
    >>> 
    >>> columns_to_keep = [
    >>>   'col_1',
    >>>   'col_4',
    >>>   'col_12'
    >>> ]
    >>> dataset = read_data_from_delta(dict_config, columns_to_keep, to_pandas=True)
  """
  
  container_name = dict_config['container_name']
  storage_account_name = dict_config['storage_account_name']
  folder_path = dict_config['folder_path']
  file_name = dict_config['file_name']
  
  full_path = f"abfss://{container_name}@{storage_account_name}.dfs.core.windows.net{folder_path}{file_name}"
  
  # spark read and select
  df = spark.read.format("delta")\
            .load(full_path)\
            .select([F.col(c).cast("string") for c in columns_to_keep])
  
  if to_pandas:
    return df.toPandas()
  else:
    return df

# COMMAND ----------

def save_pandas_to_delta(dict_config, pandas_df):
  """ function that saves a pandas dataframe to a Delta path
    Parameters:
        dict_config (dict) : dictionary with info containing storage_account_name, container_name
                             folder_path and file_name
        pandas_df (dataframe) : a pandas dataframe
    Return:
        none

    Examples:
    >>> dict_config = {
    >>>   'storage_account_name' : '****',
    >>>   'container_name' : '****',
    >>>   'folder_path' : '****',
    >>>   'file_name' : '****'
    >>> }
    >>>  save_pandas_to_delta(dict_config, df)
  """
  
  container_name = dict_config['container_name']
  storage_account_name = dict_config['storage_account_name']
  folder_path = dict_config['folder_path']
  file_name = dict_config['file_name']
  
  full_path = f"abfss://{container_name}@{storage_account_name}.dfs.core.windows.net{folder_path}{file_name}"
  
  #Create PySpark DataFrame from Pandas
  spark_df = spark.createDataFrame(pandas_df)\
                 .write.format("delta")\
                 .mode('append')\
                 .save(full_path)

# COMMAND ----------

def read_data_from_csv(dict_config, sep='|', to_pandas=False):
  """ function that retrieve data from a csv file and returns a dataframe
    Parameters:
        dict_config (dict) : dictionary with info containing storage_account_name, container_name
                             folder_path and file_name
        sep (str) : separation character in the csv file 
        to_pandas (bool): whether we want a pandas (True) or a spark (False) dataframe
    Return:
        df : pandas or spark dataframe

    Examples:
    >>> dict_config = {
    >>>   'storage_account_name' : '****',
    >>>   'container_name' : '****',
    >>>   'folder_path' : '****',
    >>>   'file_name' : '****'
    >>> }
    >>> 
    >>> dataset = read_data_from_csv(dict_config, to_pandas=True)
  """
  
  container_name = dict_config['container_name']
  storage_account_name = dict_config['storage_account_name']
  folder_path = dict_config['folder_path']
  file_name = dict_config['file_name']
  
  full_path = f"abfss://{container_name}@{storage_account_name}.dfs.core.windows.net{folder_path}{file_name}"
  
  # spark read and select
  df = spark.read.format("csv") \
        .option("charset", "utf-8") \
        .option("inferSchema", "true") \
        .option("header", "true") \
        .option("sep", '|') \
        .load(full_path)
  
  if to_pandas:
    return df.toPandas()
  else:
    return df

# COMMAND ----------

def save_pandas_to_csv(dict_config, pandas_df, delimiter='|'):
  """ function that saves a pandas dataframe to a csv file
    Parameters:
        dict_config (dict) : dictionary with info containing 
                             storage_account_name, container_name, folder_path and file_name
        pandas_df (dataframe) : a pandas dataframe
    Return:
        none

    Examples:
    >>> dict_config = {
    >>>   'storage_account_name' : '****',
    >>>   'container_name' : '****',
    >>>   'folder_path' : '****',
    >>>   'file_name' : '****'
    >>> }
    >>>  save_pandas_to_csv(dict_config, df)
  """
  
    
  container_name = dict_config['container_name']
  storage_account_name = dict_config['storage_account_name']
  folder_path = dict_config['folder_path']
  file_name = dict_config['file_name']
  
  full_path = f"abfss://{container_name}@{storage_account_name}.dfs.core.windows.net{folder_path}{file_name}"
  
  #save dataframe to file_path_CSV folder using CSV, using coalesce to get only one partition
  spark_df = spark.createDataFrame(pandas_df)\
            .coalesce(1)\
            .write.format("csv")\
            .mode('overwrite')\
            .option("header",True)\
            .option("delimiter", delimiter)\
            .save(full_path)


# COMMAND ----------

def save_sparkDF_to_delta(dict_config, df, mode='overwrite'):
  """ function that saves a pyspark dataframe to a Delta path
    Parameters:
        dict_config (dict) : dictionary with info containing storage_account_name, 
                             container_name, folder_path and file_name
        df (dataframe) : a pyspark dataframe
        mode (str) : write mode (append, overwrite)
    Return:
        none

    Examples:
    >>> dict_config = {
    >>>   'storage_account_name' : '****',
    >>>   'container_name' : '****',
    >>>   'folder_path' : '****',
    >>>   'file_name' : '****'
    >>> }
    >>>  save_sparkDF_to_delta(dict_config, df)
  """
  
  container_name = dict_config['container_name']
  storage_account_name = dict_config['storage_account_name']
  folder_path = dict_config['folder_path']
  file_name = dict_config['file_name']
  
  full_path = f"abfss://{container_name}@{storage_account_name}.dfs.core.windows.net{folder_path}{file_name}"
  
  df_write = df\
            .write.format("delta")\
            .mode('overwrite')\
            .save(full_path)
