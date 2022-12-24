# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://portalcliente.santanderconsumer.es/app/assets/img/red-logo-new.png" alt="Databricks Learning" style="width: 250px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # ML_Model 
# MAGIC 
# MAGIC ## Model_selection - HPO in pySpark
# MAGIC 
# MAGIC The process of finding the best-performing model from a set of models that were produced by different ML algorithms or different hyperparameter settings of a same algorithm is called model selection. In this set of notebooks we carry out the process of model selection by a hyperparameter tunning process. We select the best posible model by using hyperparameter optimization and training with cross-validation.
# MAGIC 
# MAGIC This notebook uses pySpark.
# MAGIC 
# MAGIC [MLlib documentation on model selection](https://spark.apache.org/docs/2.1.0/ml-tuning.html#train-validation-split)

# COMMAND ----------

# MAGIC %run ../../../../0_Includes/configuration

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Import training dataset

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

# MAGIC %md
# MAGIC ## 2. Training a classification model (with model selection)

# COMMAND ----------

train.show(5)

# COMMAND ----------

# list of categorical and numerical variables
cat_vars = [c for c in train.columns if c not in ['IM_SCORIN', target_column]]
num_vars = ['IM_SCORIN']

# set numerical to double type
for c in num_vars:
  train = train.withColumn(c, F.col(c).cast(T.DoubleType()))

# map target to 1-False (fully paid) / 6-True (default)
train = train.withColumn(target_column, F.when(F.col(target_column) == '1', False).otherwise(True))

# COMMAND ----------

# ohe categorical
string_index = [StringIndexer(inputCol=col, outputCol=col+'_indexed') for col in cat_vars]
onehot_enc = [mlOneHotEncoder(inputCol=col+'_indexed', outputCol=col+'_ohe') for col in cat_vars]
ppl_cat = mlPipeline(stages = string_index + onehot_enc)

# fit categorical transformer to train
ppl_cat_fit = ppl_cat.fit(train)

# transform cats in train 
train = ppl_cat_fit.transform(train)

# numerical
assembler = [VectorAssembler(inputCols=[col], outputCol=col+'_vec') for col in num_vars]
scale = [mlStandardScaler(inputCol=col+'_vec', outputCol=col+'_scaled') for col in num_vars]
ppl_num = mlPipeline(stages = assembler + scale)

# fit numerical transformer to train transformed
ppl_num_fit = ppl_num.fit(train)

# transform numerical in train
train = ppl_num_fit.transform(train)

trainPersisted = train.persist()

# COMMAND ----------

trainPersisted.show()

# COMMAND ----------

# MAGIC %md
# MAGIC  
# MAGIC ### 2.1 Model selection via cross-validation

# COMMAND ----------

# RF necesita que las clases sean tipo doble:
trainPersisted = trainPersisted.withColumn(target_column, F.col(target_column).cast(T.DoubleType()))

# model pipeline
assembler = VectorAssembler(inputCols=[c for c in trainPersisted.columns if '_ohe' in c or '_scaled' in c], outputCol='features')
model = mlRandomForestClassifier(featuresCol='features', labelCol=target_column)
pipe = mlPipeline(stages = [assembler, model])

# evaluator area under the PR curve
evaluator = BinaryClassificationEvaluator(rawPredictionCol='prediction', 
                                   labelCol=target_column, 
                                   metricName="areaUnderPR")

paramGrid = ParamGridBuilder() \
    .addGrid(model.numTrees, [10, 50, 100]) \
    .addGrid(model.maxDepth, [2, 5, 10]) \
    .addGrid(model.minInstancesPerNode, [1, 3, 5]) \
    .build()

crossval = CrossValidator(estimator=pipe,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=5)

# fit model to train
fittedModel = crossval.fit(trainPersisted)


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

# COMMAND ----------

# numerical variables to doble
for c in num_vars:
  test = test.withColumn(c, F.col(c).cast(T.DoubleType()))

# class in train to double type
test = test.withColumn(target_column, F.col(target_column).cast(T.DoubleType()))

# transform cats in test 
test = ppl_cat_fit.transform(test)

# transform numerical in train
test = ppl_num_fit.transform(test)

testPersisted = test.persist()

# COMMAND ----------

fittedModel.bestModel.stages

# COMMAND ----------

# use mlflow to register the model

with mlflow.start_run(run_name='RF-HPO-Spark') as run:
  
  # register categorical transformer
  mlflow.spark.log_model(ppl_cat_fit, "spark-cat-transformer")

  # register numerical transformer
  mlflow.spark.log_model(ppl_num_fit, "spark-num-transformer")
  
  # register the model
  mlflow.spark.log_model(fittedModel.bestModel.stages[-1], "spark-best-model")

  # register the model pipeline
  mlflow.spark.log_model(fittedModel, "spark-fitted-model-pipeline")
  
  # predictions on train
  predictTrain = fittedModel.transform(trainPersisted).select(target_column, "prediction")

  # predictions on test
  predictTest = fittedModel.transform(testPersisted).select(target_column, "prediction")

  # evaluate the model
  areaUnderPR = evaluator.evaluate(predictTest, {evaluator.metricName:'areaUnderPR'})
  areaUnderROC = evaluator.evaluate(predictTest, {evaluator.metricName:'areaUnderROC'})
  
  metrics_dict = {"areaUnderPR_test": areaUnderPR,
                  "areaUnderROC_test": areaUnderROC}

  mlflow.log_metrics(metrics_dict)