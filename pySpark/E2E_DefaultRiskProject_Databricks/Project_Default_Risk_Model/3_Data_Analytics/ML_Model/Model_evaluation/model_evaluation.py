# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # ML_Model 
# MAGIC 
# MAGIC ## Model_evaluation
# MAGIC 
# MAGIC Notebooks for evaluating the models produced in the previous Model_training and Model_selection steps. The evaluation tries to mimic similar production settings and for that it's very important to use an evaluation dataset (commonly known as test set) that has never been used before in previous steps. 
# MAGIC 
# MAGIC The best practice is to load a previously saved model (via MLflow) and proceed with the evaluation. 
# MAGIC 
# MAGIC Once the models have been evaluated, the recommendation is to register the model in ML-Flow in order to track the different executions, and then select the model that fits the best with the defined performance params. The selected models need to be change to production phase in ML-Flow(Local Databricks workspace).
# MAGIC 
# MAGIC This notebook will use python and pySpark indistinctly.

# COMMAND ----------

# MAGIC %run ../../../0_Includes/configuration

# COMMAND ----------

client = MlflowClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Import test set for model evaluation (the test set has never been seen before during training or model selection)

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

test.cache()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Test python models

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC **Find the models we want to evaluate in the model registery**

# COMMAND ----------

# list all registered models
for rm in client.list_registered_models():
    pprint(dict(rm), indent=4)
    

# COMMAND ----------

# search for the hpo model and the preprocessor
for mv in client.search_model_versions("name='rf-hpo'"):
    pprint(dict(mv), indent=4)
  
for mv in client.search_model_versions("name='rf-hpo-preprocessor'"):
    pprint(dict(mv), indent=4)


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC **Get the python artifacts related to the model**

# COMMAND ----------

# load the model and preprocessor from the run we like

run_id = '682b4bd4a89d451f965ec425db2ffdd2' # this is the ID of the run we are happy with

preprocessor = mlflow.sklearn.load_model(
  'runs:/{run_id}/preprocessor'.format(run_id=run_id)
)
model = mlflow.sklearn.load_model(
  'runs:/{run_id}/model'.format(run_id=run_id)
)

# print the different steps in the model pipeline
model, preprocessor

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC **Evaluate the model (this is what we should expect in production)**

# COMMAND ----------

pandas_test = test.toPandas()

# COMMAND ----------

# Define classification labels & features
labels = pandas_test['CD_CODSUBCO'] == '6' # write-off
features = pandas_test.drop(['CD_CODSUBCO'], axis=1)

# categorical and numerical variables
cat_vars = [c for c in features.columns if c != 'IM_SCORIN']
num_vars = ['IM_SCORIN']

# COMMAND ----------

# Define test classification labels & features
X_test = preprocessor.transform(features)
y_test = labels.values

# COMMAND ----------

# get predictions and metrics on test
predictions = model.predict(X_test)
(precision, recall, roc_auc) = eval_metrics(y_test, predictions)
(precision, recall, roc_auc)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Test pySpark models

# COMMAND ----------

# categorical and numerical variables
cat_vars = [c for c in test.columns if c != 'IM_SCORIN']
num_vars = ['IM_SCORIN']

# numerical variables to doble
for c in num_vars:
  test = test.withColumn(c, F.col(c).cast(T.DoubleType()))

# class in train to double type
test = test.withColumn(target_column, F.col(target_column).cast(T.DoubleType()))

# COMMAND ----------

# search for the hpo model and the preprocessor
for mv in client.search_model_versions("name='spark-best-model'"):
    pprint(dict(mv), indent=4)
  

# COMMAND ----------

# set the run id
run_id = '9b70c966b332422e81e026c8a18e0454'

# categorical transformer
cat_transformer = mlflow.spark.load_model(model_uri='runs:/{run_id}/spark-cat-transformer'.format(run_id=run_id))

# numerical transformer
num_transformer = mlflow.spark.load_model(model_uri='runs:/{run_id}/spark-num-transformer'.format(run_id=run_id))

# model
model = mlflow.spark.load_model(model_uri='runs:/{run_id}/spark-best-model'.format(run_id=run_id))

# pipeline
pipe = mlflow.spark.load_model(model_uri='runs:/{run_id}/spark-fitted-model-pipeline'.format(run_id=run_id))



# COMMAND ----------

# transform cats in test 
test = cat_transformer.transform(test)

# transform numerical in train
test = num_transformer.transform(test)

testPersisted = test.persist()

# COMMAND ----------

# predictions on test
predictTest = pipe.transform(testPersisted).select(target_column, "prediction")

# evaluate the model
# evaluator area under the PR curve
evaluator = BinaryClassificationEvaluator(rawPredictionCol='prediction', 
                                   labelCol=target_column)

areaUnderPR = evaluator.evaluate(predictTest, {evaluator.metricName:'areaUnderPR'})
areaUnderROC = evaluator.evaluate(predictTest, {evaluator.metricName:'areaUnderROC'})


# COMMAND ----------

(areaUnderPR, areaUnderROC)
