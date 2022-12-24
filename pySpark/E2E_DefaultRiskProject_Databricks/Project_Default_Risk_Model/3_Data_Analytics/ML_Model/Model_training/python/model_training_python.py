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
# MAGIC In this notebook we train an ML model using the training dataset created in previous notebooks. This notebook contains simple model training without hyperparameter optimization.
# MAGIC 
# MAGIC This notebook uses pure python libraries.

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

# aquí estamos haciendo implícitamente un .toPandas() y llevando todo el dato al driver
# este paso puede ser problemático para datos masivos
pandas_df = read_data_from_delta(dict_config, 
                               columns_to_keep=[c for c in columns_to_keep if c!='ID_CODCONTR'], 
                               to_pandas=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Entrenamiento de un modelo de clasificación

# COMMAND ----------

pandas_df.head()

# COMMAND ----------

# Define classification labels & features
labels = pandas_df['CD_CODSUBCO'] == '6' # write-off
features = pandas_df.drop(['CD_CODSUBCO'], axis=1)

# We do holdout validation with a 80/20 train-validation split
X_train, X_val, y_train, y_val = train_test_split(
  features,
  labels,
  test_size=0.2,
  random_state=RANDOM_SEED
)

# COMMAND ----------

X_train

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.1 Entrenamiento y tracking con MLflow

# COMMAND ----------

# MAGIC %md
# MAGIC Un modelo de ML puede ser desde muy simple a muy complejo. 
# MAGIC 
# MAGIC En este caso, vamos a concatenar en nuestro modelo diversos pasos en una pipeline de scikit encapsulada en la función `model_training` definida en el notebook `ML_UDFS_utilities`.
# MAGIC 
# MAGIC Esta pipeline aplica OHE a las features categóticas, un StandardScaler a las numéricas y además hace un resampleo SMOTE para balancear las clases.
# MAGIC 
# MAGIC El modelo es evaluado en el dataset de validación usando la función `eval_metrics` definida tambien en ese notebook.

# COMMAND ----------

cat_vars = [c for c in features.columns if c != 'IM_SCORIN']
num_vars = ['IM_SCORIN']

# COMMAND ----------

# Start a new run and assign a run_name for future reference
with mlflow.start_run(run_name='logistic_regression') as run:
  
    # set the estimator (the ML algo we are going to use)
    estimator = LogisticRegression()

    # register model hyperparameters
    mlflow.log_params({"C": estimator.C, "penalty": estimator.penalty})

    # fit the pipe: encoding + sampling + estimator
    model_pipe = model_training(estimator, X_train, y_train, cat_vars, num_vars, balance=True)

    # get predictions and metrics
    predictions = model_pipe.predict(X_val)
    (precision, recall, roc_auc) = eval_metrics(y_val, predictions)
    
    # register metrics and fitted model
    mlflow.log_metrics({"precision": precision, 
                        "recall": recall, 
                        "roc_auc": roc_auc})
    mlflow.sklearn.log_model(model_pipe, "model")

    # print performance metrics on test
    print("precision: {}\nrecall: {}\nroc_auc: {}".format(precision, recall, roc_auc))

# COMMAND ----------

# Start a new run and assign a run_name for future reference
with mlflow.start_run(run_name='random_forest') as run:
  
  # set the estimator (the ML algo we are going to use)
  estimator = RandomForestClassifier()
  
  # register model hyperparameters
  mlflow.log_params({"n_estimators": estimator.n_estimators, 
                     "max_depth": estimator.max_depth})
  
  # fit the pipe: encoding + sampling + estimator
  model_pipe = model_training(estimator, X_train, y_train, cat_vars, num_vars, balance=True)
  
  # get predictions and metrics
  predictions = model_pipe.predict(X_val)
  (precision, recall, roc_auc) = eval_metrics(y_val, predictions)
    
  # register metrics and fitted model
  mlflow.log_metrics({"precision": precision, 
                      "recall": recall, 
                      "roc_auc": roc_auc})
  mlflow.sklearn.log_model(model_pipe, "model")

  # print performance metrics on test
  print("precision: {}\nrecall: {}\nroc_auc: {}".format(precision, recall, roc_auc))

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC #### Cómo podemos ver los distintos modelos ejecutados?
# MAGIC 
# MAGIC Ir al icono **Experiment** en la parte superior derecha de esta ventana para ver los distintos experimentos que se han ejecutado.
# MAGIC 
# MAGIC <img width="350" src="https://docs.databricks.com/_static/images/mlflow/quickstart/experiment-sidebar-icons.png"/>
# MAGIC 
# MAGIC Podemos hace click en la página del experimento para ver más detalles ([Azure](https://docs.microsoft.com/azure/databricks/applications/mlflow/tracking#notebook-experiments)). Esta página permite comparar diferentes experimentos.
# MAGIC 
# MAGIC <img width="800" src="https://docs.databricks.com/_static/images/mlflow/quickstart/compare-runs.png"/>

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Cómo podemos cargar un modelo ya registrado How to load models previously registered
# MAGIC 
# MAGIC Podemos acceder a los resultaso de una ejecución concreta mediante la API de MLflow, usando el ID del modelo registrado. 
# MAGIC 
# MAGIC En la siguiente celda comparamos las predicciones del último modelo registrado descargandolo del repositorio de modelos. 

# COMMAND ----------

model_loaded = mlflow.pyfunc.load_model(
  'runs:/{run_id}/model'.format(
    run_id=run.info.run_id
  )
)

predictions_loaded = model_loaded.predict(X_val)
predictions_original = model_pipe.predict(X_val)

# The loaded model should match the original
np.array_equal(predictions_loaded, predictions_original)

# COMMAND ----------

# MAGIC %md
# MAGIC Usando la API de MLflow podemos descargarnos todos los artefactos asociados a un modelo, para que este sea reproducible en cualquier plataforma.
# MAGIC 
# MAGIC Nótese que estamos usando la función `fetch_logged_data` definida en `ML_UDFS_utilities`.

# COMMAND ----------

#### se puede especificar a mano:
####run_id = '<<<<>>>>>'

#### o bien obtener el id programáticamente:
run_id = run.info.run_id
print("Logged data and model in run: {}".format(run_id))

# fetch logged data
params, metrics, tags, artifacts = fetch_logged_data(run_id)

print('\n------ model hyperparameters ------\n')
pprint(params)

print('\n------ model performance metrics ------\n')
pprint(metrics)

print('\n------ model tags ------\n')
pprint(tags)

print('\n------ mlflow artifacts ------\n')
pprint(artifacts)


# COMMAND ----------

# MAGIC %md
# MAGIC Uno de estos artefactos es el "pickle" del modelo, que puede ser usado igualmente posteriormente.

# COMMAND ----------

client = mlflow.tracking.MlflowClient()
tmp_path = client.download_artifacts(run_id=run_id, path='model/model.pkl')
f = open(tmp_path,'rb')
model_loaded = pickle.load(f)

predictions_loaded = model_loaded.predict(X_val)
predictions_original = model_pipe.predict(X_val)

# The loaded model should match the original
np.array_equal(predictions_loaded, predictions_original)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Model serving via REST API  
# MAGIC 
# MAGIC - Ir a la pestaña de experimentos
# MAGIC - Registrar el modelo
# MAGIC - Una vez registrado el modelo puede servirse en batch o on-line tras generar una REST API
