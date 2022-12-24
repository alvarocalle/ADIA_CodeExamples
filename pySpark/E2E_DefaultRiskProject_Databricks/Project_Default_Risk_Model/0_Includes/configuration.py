# Databricks notebook source
# MAGIC %md
# MAGIC #### Configuration  
# MAGIC 
# MAGIC - Notebook used to orchestrate all configuration notebooks.
# MAGIC 
# MAGIC - Pay attention to the run order.

# COMMAND ----------

# DBTITLE 1,Define Global Variables and Constants
# Not applicable for this case
#%run ./Variables_Path_Utilities

# COMMAND ----------

# DBTITLE 1,Define Data Paths
# MAGIC %run ./Utilities/data_path_utilities

# COMMAND ----------

# DBTITLE 1,Define Hive Metastore configuration
# Not applicable for this case
####%run ./Utilities/hive_metastore_utilities

# COMMAND ----------

# DBTITLE 1,Define Project Enviroment configuration
# MAGIC %run ./Utilities/environment_configuration_utilities

# COMMAND ----------

# DBTITLE 1,Define Project ETL User Defined Functions Utilites
# MAGIC %run ./Utilities/ETL_UDFS_utilities

# COMMAND ----------

# DBTITLE 1,Define Project ML User Defined Functions Utilites
# MAGIC %run ./Utilities/ML_UDFS_utilities

# COMMAND ----------

# DBTITLE 1,Define Project Test Utilites
# Not applicable for this case
#%run ./Test_Utilities