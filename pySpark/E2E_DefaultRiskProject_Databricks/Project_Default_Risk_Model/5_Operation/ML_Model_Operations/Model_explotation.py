# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # ML_Model_Operations
# MAGIC 
# MAGIC ## Model_explotation
# MAGIC 
# MAGIC In this notebook we manage possible the data exports comming from the trained models, in case our model(s) generates data that we need to store in Azure Data Storage. 
# MAGIC 
# MAGIC The exports have to be saved in a `/ML_output` folder of the storage account.

# COMMAND ----------

###%run ../../0_Includes/configuration

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Steps to follow:
# MAGIC 
# MAGIC - Set GLOBAL parameter for tracking (like dates, flags, etc).
# MAGIC - Load the registered models (you can also load from pickle if needed).
# MAGIC - Create/register necesary tables in hive matastore (through spark).
# MAGIC - Execute model on necessary tables (dataframes).
# MAGIC - Export results in a suitable form.

# COMMAND ----------

