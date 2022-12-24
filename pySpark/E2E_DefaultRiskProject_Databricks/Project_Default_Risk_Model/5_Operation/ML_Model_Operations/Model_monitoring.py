# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # ML_Model_Operations
# MAGIC 
# MAGIC ## Model_monitoring
# MAGIC 
# MAGIC Notebook for analyzing **early signals** or threshold warnings (e.g. accuracy under a specific value), from the outcomes of the system implemented in the test phase. Performance can also be monitored by comparing the outcomes of the ML model with those of the previous system running in parallel.
# MAGIC 
# MAGIC - We assume we have a notebook that is running the different in-production models in batch. The results from inference are saved in (hive) tables.
# MAGIC - This notebook tracks these metrics and may apply different methods to detect drift.

# COMMAND ----------

###%run ../../0_Includes/configuration

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Steps to follow:
# MAGIC 
# MAGIC - Load GLOBAL thresholds for tracking (accuracy, RMSE, ...) and confidence limits.
# MAGIC - Load the traking metrics tables.
# MAGIC - Apply necessary statistical tests to detect drifts.

# COMMAND ----------

