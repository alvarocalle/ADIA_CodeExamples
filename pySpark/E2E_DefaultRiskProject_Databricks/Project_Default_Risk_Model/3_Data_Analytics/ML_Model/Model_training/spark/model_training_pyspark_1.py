# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # ML_Model 
# MAGIC 
# MAGIC ## Model_training
# MAGIC 
# MAGIC ### Simple model
# MAGIC 
# MAGIC In this notebook we train an ML model using the training dataset created in previous notebooks. This notebook contains simple model training without hyperparameter optimization.
# MAGIC 
# MAGIC This notebook uses pySpark libraries.
# MAGIC 
# MAGIC [MLlib API docs](https://spark.apache.org/docs/2.1.0/ml-guide.html)

# COMMAND ----------

# MAGIC %run ../../../../0_Includes/configuration

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Importamos el dataset de training

# COMMAND ----------

dict_config = {
  'storage_account_name' : storage_account_name,
  'container_name' : container_name,
  'folder_path' : folder_path_OUT,
  'file_name' : '/train' 
}

# read spark dataframe
train = read_data_from_delta(dict_config,
                             columns_to_keep=[c for c in columns_to_keep if c!='ID_CODCONTR'],
                             to_pandas=False)

# we save it in memory as we are performing different operations on it
train.cache()

# COMMAND ----------

train.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Entrenamiento de un modelo de clasificación

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.1 Entrenamiento simple con Spark

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC En este notebook vamos a ver el caso de entrenamiento de un modelo con pySpark usando MLlib API. No todas las herramientas de que disponemos en python están disponibles para ser usadas con spark y a veces tocará hacer desarrollos propios. La ventaja será que los modelos serán totalmente escalables.

# COMMAND ----------

# empezamos como antes definiendo nuestra lista de variables numéricas y categóricas
cat_vars = [c for c in train.columns if c != 'IM_SCORIN']
num_vars = ['IM_SCORIN']

# nos aseguramos convertir las variables numericas a doble
for c in num_vars:
  train = train.withColumn(c, F.col(c).cast(T.DoubleType()))

# mapeamos la variable target a 1-False (fully paid) / 6-True (default)
train = train.withColumn(target_column, F.when(F.col(target_column) == '1', False).otherwise(True))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Vamos a empezar creando un modelo sencillo.
# MAGIC 
# MAGIC Siempre que se usa MLlib, todos los inputs de los algoritmos de machine learning (con algunas excepciones) en Spark deben ser tipe Double (para los labels) y tipo Vector(Double) (para las features). Para ello siempre hay que crear un vector de features usando [`VectorAssembler`](https://spark.apache.org/docs/3.1.3/api/python/reference/api/pyspark.ml.feature.VectorAssembler.html). 
# MAGIC 
# MAGIC Cuando queramos crear un primer prototipo, hay una forma rápida de conseguir transformar el dataset en una forma que sea usable por MLlib, mediante [`RFormula`](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.RFormula.html), una clase que funciona igual que como se crean los modelos lineales en R. En este caso, vamos a usarlo para trasformar el dataset únicamente. 

# COMMAND ----------

train.printSchema()


# COMMAND ----------

print("{target} ~ .".format(target=target_column))

# COMMAND ----------

# build the model
rForm = RFormula(formula="{target} ~ .".format(target=target_column))
lr = LogisticRegression().setLabelCol("label").setFeaturesCol("features")

fittedRF = rForm.fit(train)
preparedDF = fittedRF.transform(train)
preparedDF.show()

# COMMAND ----------

# 80/20 train/validation split
labels = (0.0, 1.0)
train_fraction = 0.8
train_, validation_ = sparkDF_stratified_train_test_split(preparedDF, 
                                                  target_column="label", 
                                                  labels=labels, 
                                                  train_fraction=train_fraction, 
                                                  random_state=RANDOM_SEED)

# COMMAND ----------

train_.count(), validation_.count() 

# COMMAND ----------

# fit model to train_
fittedLR = lr.fit(train_)

# COMMAND ----------

# predictions on train_
fittedLR.transform(train_).select("label", "prediction").show()

# COMMAND ----------

# predictions on validation_
fittedLR.transform(validation_).select("label", "prediction").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2 Traking del modelo con MLflow

# COMMAND ----------

mlflow.spark.log_model(fittedLR, "spark-model")


# COMMAND ----------

# Load model as a Spark UDF. Override result_type if the model does not return double values.
loaded_model = mlflow.spark.load_model('dbfs:/databricks/mlflow-tracking/1703014099768387/c083752c6b274c53bee79d0edc886334/artifacts/spark-model')

# COMMAND ----------

loaded_model.transform(validation_).select("label", "prediction").show()
