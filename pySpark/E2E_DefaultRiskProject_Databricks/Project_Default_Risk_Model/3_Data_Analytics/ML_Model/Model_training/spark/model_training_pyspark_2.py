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
# MAGIC ### A more sophisticated model
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
df = read_data_from_delta(dict_config,
                             columns_to_keep=[c for c in columns_to_keep if c!='ID_CODCONTR'],
                             to_pandas=False)

# we save it in memory as we are performing different operations on it
df.cache()

# COMMAND ----------

df.printSchema()

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
cat_vars = [c for c in df.columns if c not in ['IM_SCORIN', target_column]]
num_vars = ['IM_SCORIN']

# nos aseguramos convertir las variables numericas a doble
for c in num_vars:
  df = df.withColumn(c, F.col(c).cast(T.DoubleType()))

# mapeamos la variable target a 1-False (fully paid) / 6-True (default)
df = df.withColumn(target_column, F.when(F.col(target_column) == '1', False).otherwise(True))

# COMMAND ----------

[col for col in cat_vars]

# COMMAND ----------

# ohe categorical
string_index = [StringIndexer(inputCol=col, outputCol=col+'_indexed') for col in cat_vars]
onehot_enc = [mlOneHotEncoder(inputCol=col+'_indexed', outputCol=col+'_ohe') for col in cat_vars]
ppl_cat = mlPipeline(stages = string_index + onehot_enc)
df_trans = ppl_cat.fit(df).transform(df)

# numerical
assembler = [VectorAssembler(inputCols=[col], outputCol=col+'_vec') for col in num_vars]
scale = [mlStandardScaler(inputCol=col+'_vec', outputCol=col+'_scaled') for col in num_vars]
pipe = mlPipeline(stages = assembler + scale)
df_scale = pipe.fit(df_trans).transform(df_trans)
dfPersisted = df_scale.persist()

# COMMAND ----------

dfPersisted.show(5)

# COMMAND ----------

# RF necesita que las clases sean tipo doble:
dfPersisted = dfPersisted.withColumn(target_column, F.col(target_column).cast(T.DoubleType()))

# Hacemos un 80/20 split y guardamos los datasets en Delta
labels = (1.0, 6.0)
train_fraction = 0.8
train, valid = sparkDF_stratified_train_test_split(dfPersisted, 
                                                  target_column=target_column, 
                                                  labels=labels, 
                                                  train_fraction=train_fraction, 
                                                  random_state=RANDOM_SEED)

# COMMAND ----------

assembler = VectorAssembler(inputCols=[c for c in dfPersisted.columns if '_ohe' in c or '_scaled' in c], outputCol='features')
model = RandomForestClassifier(featuresCol='features', labelCol=target_column)
pipe = Pipeline(stages = [assembler, model])

# COMMAND ----------

# Compute the area under the PR curve
pr = BinaryClassificationEvaluator(rawPredictionCol='prediction', 
                                   labelCol=target_column, 
                                   metricName="areaUnderPR")

# COMMAND ----------

with mlflow.start_run(run_name='random_forest') as run:
  
  # fit model to train
  fittedModel = pipe.fit(train)
  
  # predictions on train
  predictTrain = fittedModel.transform(train).select(target_column, "prediction")
  
  # predictions on validation
  predictVal = fittedModel.transform(valid).select(target_column, "prediction")
  
  # evaluate the model
  areaUnderPR = pr.evaluate(predictVal)
  
  # register the model
  mlflow.spark.log_model(fittedModel, "spark-model")

# COMMAND ----------

# latest logged model
logged_model = 'runs:/{run_id}/spark-model'.format(run_id=run.info.run_id)

# COMMAND ----------

# inference on pySpark dataframes
loaded_model = mlflow.spark.load_model(model_uri=logged_model)
loaded_model.transform(valid).select(target_column, "prediction").show()

# COMMAND ----------

# inference on pandas
loaded_model = mlflow.pyfunc.load_model(logged_model)
loaded_model.predict(valid.toPandas())
