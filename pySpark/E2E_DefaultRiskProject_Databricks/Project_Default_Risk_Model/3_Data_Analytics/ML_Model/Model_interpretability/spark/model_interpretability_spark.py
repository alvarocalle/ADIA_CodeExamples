# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # ML_Model 
# MAGIC 
# MAGIC ## Model_interpretability
# MAGIC 
# MAGIC ### How to use shap with ML models

# COMMAND ----------

# MAGIC %run ../../../../0_Includes/configuration

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 1.- Import datasets and saved models

# COMMAND ----------

# MAGIC %md
# MAGIC Importamos un dataset sobre el que ejecutar nuestro modelo e interpretar los resultados. En este caso vamos a usar el dataset de test.

# COMMAND ----------

# MAGIC %md
# MAGIC **Load dataset**

# COMMAND ----------

dict_config = {
  'storage_account_name' : storage_account_name,
  'container_name' : container_name,
  'folder_path' : folder_path_OUT,
  'file_name' : '/test' 
}
# read spark dataframe
test = read_data_from_delta(dict_config,
                             columns_to_keep=[c for c in columns_to_keep if c!='ID_CODCONTR'],
                             to_pandas=False)

# we save it in memory as we are performing different operations on it
test.cache()

# categorical and numerical variables
cat_vars = [c for c in test.columns if c != 'IM_SCORIN']
num_vars = ['IM_SCORIN']

# numerical variables to doble
for c in num_vars:
  test = test.withColumn(c, F.col(c).cast(T.DoubleType()))

# class in train to double type
test = test.withColumn(target_column, F.col(target_column).cast(T.DoubleType()))

# COMMAND ----------

test.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **Load data transformers and ml model**

# COMMAND ----------

# set the run id
run_id = '6bf8657bc0dd430d9ef2b0461a28ae89'

# categorical transformer
cat_transformer = mlflow.spark.load_model(model_uri='runs:/{run_id}/spark-cat-transformer'.format(run_id=run_id))

# numerical transformer
num_transformer = mlflow.spark.load_model(model_uri='runs:/{run_id}/spark-num-transformer'.format(run_id=run_id))

# model
model = mlflow.spark.load_model(model_uri='runs:/{run_id}/spark-best-model'.format(run_id=run_id))

# COMMAND ----------

# transform cats in test 
test = cat_transformer.transform(test)

# transform numerical in train
test = num_transformer.transform(test)

testPersisted = test.persist()

# COMMAND ----------

testPersisted.show()

# COMMAND ----------

model.stages[0]

# COMMAND ----------

# MAGIC %md
# MAGIC **High Concurrency cluster with credential pass through enabled (Error)**
# MAGIC 
# MAGIC This error shows up with some library methods when using High Concurrency cluster with credential pass through enabled. If that is your scenario a work around that may be an option is to use a different cluster mode.
# MAGIC 
# MAGIC https://stackoverflow.com/questions/55427770/error-running-spark-on-databricks-constructor-public-xxx-is-not-whitelisted

# COMMAND ----------


# Define SHAP explainer.
explainer = shap.TreeExplainer( model.stages[0] )

# COMMAND ----------

shap_values = explainer.shap_values( testPersisted )
exp_values = explainer.expected_value
