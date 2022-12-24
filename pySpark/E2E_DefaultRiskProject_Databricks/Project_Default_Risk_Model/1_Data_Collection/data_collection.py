# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Data collection
# MAGIC 
# MAGIC - Notebooks that load the data from the different data sources (e.g. databases, data warehouses, data lakes) and/or external sources (e.g. external data providers, the internet). 
# MAGIC 
# MAGIC - The data is not transformed is loaded into bronze delta tables in raw. The orchestration can be achieved using a notebook bronze_orchestation.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Nota:
# MAGIC 
# MAGIC Para este caso de uso la carga es sencilla, ya que se ha preparado en AZ Storage un CSV con el dataset de análisis.
