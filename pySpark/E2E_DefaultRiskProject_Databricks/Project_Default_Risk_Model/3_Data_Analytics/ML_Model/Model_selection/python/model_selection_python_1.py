# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # ML_Model 
# MAGIC 
# MAGIC ## Model_selection - HPO in single node
# MAGIC 
# MAGIC The process of finding the best-performing model from a set of models that were produced by different ML algorithms or different hyperparameter settings of a same algorithm is called model selection. In this set of notebooks we carry out the process of model selection by a hyperparameter tunning process. We select the best posible model by using hyperparameter optimization and training with cross-validation.
# MAGIC 
# MAGIC This notebook uses pure python libraries.

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

# aquí estamos haciendo implícitamente un .toPandas() y llevando todo el dato al driver
# este paso puede ser problemático para datos masivos
train = read_data_from_delta(dict_config, 
                               columns_to_keep=[c for c in columns_to_keep if c!='ID_CODCONTR'], 
                               to_pandas=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Training a classification model (with model selection)
# MAGIC 
# MAGIC Supongamos que queremos hacer un ajuste de hiperparámetros. El ajuste de hiperparámetros implica la evaluación de múltiples modelos usando distintas técnicas, búsqueda en grid (GridSearch), búsqueda aleatoria (RandomSearch), o algoritmos más sofisticados (por ejemplo optimización bayesiana). Además podemos estudiar las curvas ROC, precision_recall, y otras métricas. El estudio puede ser tan exhaustivo como queramos y podemos registrar todo como artefactos en mlflow.
# MAGIC 
# MAGIC Aquí vamos a usar implementaciones de sklearn que permiten esta búsqueda de hiperparámetros.

# COMMAND ----------

train.head()

# COMMAND ----------

# Define classification labels & features
labels = train['CD_CODSUBCO'] == '6' # write-off
features = train.drop(['CD_CODSUBCO'], axis=1)

# COMMAND ----------

cat_vars = [c for c in features.columns if c != 'IM_SCORIN']
num_vars = ['IM_SCORIN']

# COMMAND ----------

# fit pre-processor to train set
preprocessor = preprocessing(features, cat_vars, num_vars)


# COMMAND ----------

X_train = preprocessor.transform(features)
y_train = labels.values

# COMMAND ----------

# model selection with 5-fold stratified CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=5)

# COMMAND ----------

lr = LogisticRegression(random_state=RANDOM_SEED, class_weight="balanced")
parameters = {"C": (0.0001, 0.001, 0.01, 0.1, 1, 10)}
model = GridSearchCV(estimator=lr, param_grid=parameters, cv=skf, refit=True, n_jobs=-1, scoring="roc_auc")
model.fit(X_train, y_train)

# COMMAND ----------

model.best_params_, model.best_score_

# COMMAND ----------

rfc = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, class_weight="balanced")
parameters = {'max_features': [4, 7, 10, 13], 
              'min_samples_leaf': [1, 3, 5, 7], 
              'max_depth': [5, 10, 15, 20]}
model = GridSearchCV(estimator=rfc, param_grid=parameters, cv=skf, refit=True, n_jobs=-1, scoring="roc_auc")
model.fit(X_train, y_train)

# COMMAND ----------

model.best_params_, model.best_score_

# COMMAND ----------

# MAGIC %md
# MAGIC Get the best model found by GridSearch and train it in the whole training set with the best hyperparameters. Notice, we have obtained the best model (model selection) using 5-fold CV. Now we use the whole dataset and register that model in mlflow.
# MAGIC 
# MAGIC Finally, in order to have an idea of the future performance we test the model on unseen test set. This can be done here to register metrics in mlflow together with the experiment, but there is a dedicated folder for model evaluation where the unseen test set must be used to assess model perfoemance in production.

# COMMAND ----------

dict_config = {
  'storage_account_name' : storage_account_name,
  'container_name' : container_name,
  'folder_path' : folder_path_OUT,
  'file_name' : '/test' 
}

# aquí estamos haciendo implícitamente un .toPandas() y llevando todo el dato al driver
# este paso puede ser problemático para datos masivos
test = read_data_from_delta(dict_config, 
                               columns_to_keep=[c for c in columns_to_keep if c!='ID_CODCONTR'], 
                               to_pandas=True)

# COMMAND ----------

# MAGIC %md
# MAGIC **IMPORTANT:** fit preprocesor to train **not** to test set. Use fitted preprocessor to transform the test.

# COMMAND ----------

# Define test classification labels & features
X_test = preprocessor.transform(test.drop(['CD_CODSUBCO'], axis=1))
y_test = (test['CD_CODSUBCO'] == '6').values


# COMMAND ----------

X_train.shape, X_test.shape

# COMMAND ----------

# select model by seting hyperparameters
params = {'n_estimators': 100, 'random_state': RANDOM_SEED, 'class_weight': 'balanced', 'max_depth': 20, 'max_features': 13, 'min_samples_leaf': 1}

# COMMAND ----------

#Create session MLFlow just to register the best model

with mlflow.start_run(run_name="RF-HPO-python") as run:
  
  run_id = run.info.run_id
  mlflow.log_params(params)

  # select model
  model = RandomForestClassifier(**params)

  # fit model to whole train (once we know what are the best hyperparameters)
  model.fit(X_train, y_train)
  
  # get predictions and metrics on test
  predictions = model.predict(X_test)
  (precision, recall, roc_auc) = eval_metrics(y_test, predictions)
    
  # register metrics and fitted model
  mlflow.log_metrics({"precision": precision, 
                      "recall": recall, 
                      "roc_auc": roc_auc})
  
    
  # register the fitted transformer (to prepare future data for the ml model)
  mlflow.sklearn.log_model(preprocessor, "preprocessor")
  
  # register the best performing model according to HPO
  mlflow.sklearn.log_model(model, "model")

  # print performance metrics on test
  print("precision: {}\nrecall: {}\nroc_auc: {}".format(precision, recall, roc_auc))
    

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Search runs to retrieve the best model
# MAGIC 
# MAGIC Because all of the runs are tracked by MLflow, you can retrieve the metrics and parameters for the best run using the MLflow search runs API to find the tuning run with the highest test auc.
# MAGIC 
# MAGIC This tuned model should perform better than the simpler models trained before.

# COMMAND ----------

# Sort runs by their test auc; in case of ties, use the most recent run
best_run = mlflow.search_runs(
  order_by=['metrics.roc_auc DESC', 'start_time DESC'],
  max_results=10,
).iloc[0]

print('Best Run')
print('AUC: {}'.format(best_run["metrics.roc_auc"]))
print('Num Estimators: {}'.format(best_run["params.n_estimators"]))
print('Max Depth: {}'.format(best_run["params.max_depth"]))

best_model_pyfunc = mlflow.pyfunc.load_model(
  'runs:/{run_id}/model'.format(
    run_id=best_run.run_id
  )
)

best_model_predictions = best_model_pyfunc.predict(X_test[:5])
print("Test Predictions: {}".format(best_model_predictions))
