# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # ML_Model 
# MAGIC ## Model_case_table
# MAGIC 
# MAGIC In this notebook the data scientist can use data from silver and gold delta table, the process englobe data cleansing and feature engineering and the result of the process is a table that is used for the model training
# MAGIC 
# MAGIC The output of this notebook should be two tables: training + test
# MAGIC 
# MAGIC - Training: used for model training, optimization and model selection
# MAGIC - Test: Used for model evaluation at the end

# COMMAND ----------

# MAGIC %run ../../../0_Includes/configuration

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Importamos el raw dataset guardado en el Delta lake y aplicamos el preprocesado necesario

# COMMAND ----------

dict_config = {
  'storage_account_name' : storage_account_name,
  'container_name' : container_name,
  'folder_path' : folder_path_IN + csv_folder,
  'file_name' : file_name 
}

# read the data and store it in a pyspark dataframe
df = read_data_from_csv(dict_config, sep='|', to_pandas=False)

# COMMAND ----------

# apply pre-processing to the dataset
df = cleanssing(df)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### 2. Creación de las `model_case_tables` : train / test split

# COMMAND ----------

# Hacemos un 80/20 split y guardamos los datasets en Delta
labels = ('1', '6')
train_fraction = 0.8
train, test = sparkDF_stratified_train_test_split(df, 
                                                  target_column=target_column, 
                                                  labels=labels, 
                                                  train_fraction=train_fraction, 
                                                  random_state=RANDOM_SEED)

# COMMAND ----------

df.count(), train.count(), test.count()

# COMMAND ----------

dict_config = {
  'storage_account_name' : storage_account_name,
  'container_name' : container_name,
  'folder_path' : folder_path_OUT,
  'file_name' : '/train' 
}
save_sparkDF_to_delta(dict_config, train)

# COMMAND ----------

dict_config = {
  'storage_account_name' : storage_account_name,
  'container_name' : container_name,
  'folder_path' : folder_path_OUT,
  'file_name' : '/test' 
}
save_sparkDF_to_delta(dict_config, test)
