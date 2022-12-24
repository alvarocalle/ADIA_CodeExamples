# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # ML_Model 
# MAGIC 
# MAGIC ## Model_selection - HPO in a cluster 
# MAGIC   [THIS NOTEBOOK DOESN'T WORK BECAUSE CLUSTER CONFIGURATION]
# MAGIC 
# MAGIC The process of finding the best-performing model from a set of models that were produced by different ML algorithms or different hyperparameter settings of a same algorithm is called model selection. In this set of notebooks we carry out the process of model selection by a hyperparameter tunning process. We select the best posible model by using hyperparameter optimization and training with cross-validation.
# MAGIC 
# MAGIC This notebook uses pure python libraries.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC **Problems with high concurrency clusters with passthroug enabled**
# MAGIC 
# MAGIC - https://stackoverflow.com/questions/55427770/error-running-spark-on-databricks-constructor-public-xxx-is-not-whitelisted
# MAGIC - https://community.databricks.com/s/question/0D53f00001OFuWLCA1/can-you-help-with-this-error-please-issue-when-using-a-new-high-concurrency-cluster

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
pandas_df = read_data_from_delta(dict_config, 
                               columns_to_keep=[c for c in columns_to_keep if c!='ID_CODCONTR'], 
                               to_pandas=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Training a classification model (with model selection)

# COMMAND ----------

pandas_df.head()

# COMMAND ----------

# Define classification labels & features
labels = pandas_df['CD_CODSUBCO'] == '6' # write-off
features = pandas_df.drop(['CD_CODSUBCO'], axis=1)

X_train = features.values
y_train = labels.values

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ### 2.2 Hyperparameter Tuning
# MAGIC 
# MAGIC Supongamos que queremos hacer un ajuste de hiperparámetros. El ajuste de hiperparámetros implica la evaluación de múltiples modelos usando distintas técnicas, búsqueda en grid (GridSearch), búsqueda aleatoria (RandomSearch), o algoritmos más sofisticados (por ejemplo optimización bayesiana). 
# MAGIC 
# MAGIC Aquí vamos a usar la librería [Hyperopt](https://github.com/hyperopt/hyperopt), que es una librería que permite que esta búsqueda de hiperparámetros sea distribuida. Gracias a la infraestructura de Databricks, esta trabajo distribuido es directo.
# MAGIC 
# MAGIC Podemos usar `Hyperopt` con `SparkTrials` para cambiar hiperparámetros y entrenar múltiples modelos en paralelo. Esto reduce considerablemente el tiempo requerido para optimizar nuestro modelo. MLflow permite traquear los distintos modelos ya que está integrado con Hyperopt.
# MAGIC 
# MAGIC Vamos a asumir que queremos optimizar un [scikit-learn Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) y queremos optimizar dos de sus hiperparámetros:
# MAGIC 
# MAGIC - `n_estimators`: el número de árboles que va a tener el bosque.
# MAGIC - `max_depth`: la máxima profundidad del árbol.
# MAGIC 
# MAGIC El resto de hiperparámetros los dejamos con sus valores por defecto.

# COMMAND ----------

# Define the search space to explore
# In this case we are assuming a RF model
search_space = {
  'n_estimators': scope.int(hp.quniform('n_estimators', 20, 1000, 1)),
  'max_depth': scope.int(hp.quniform('max_depth', 2, 5, 1))
}

# SparkTrials distributes the tuning using Spark workers
# Greater parallelism speeds processing, but each hyperparameter trial has less information from other trials
# On smaller clusters or Databricks Community Edition try setting parallelism=2
spark_trials = SparkTrials(
  parallelism=2
)

# Use hyperopt to find the parameters yielding the highest AUC
with mlflow.start_run(run_name='rf_hyperopt') as run:
  
  best_params = fmin(
                  fn=train_model_with_hpo, 
                  space=search_space, 
                  algo=tpe.suggest, 
                  max_evals=32,
                  trials=spark_trials)

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
  order_by=['metrics.test_auc DESC', 'start_time DESC'],
  max_results=10,
).iloc[0]

print('Best Run')
print('AUC: {}'.format(best_run["metrics.test_auc"]))
print('Num Estimators: {}'.format(best_run["params.n_estimators"]))
print('Max Depth: {}'.format(best_run["params.max_depth"]))
print('Learning Rate: {}'.format(best_run["params.learning_rate"]))

best_model_pyfunc = mlflow.pyfunc.load_model(
  'runs:/{run_id}/model'.format(
    run_id=best_run.run_id
  )
)

best_model_predictions = best_model_pyfunc.predict(X_test[:5])
print("Test Predictions: {}".format(best_model_predictions))

# COMMAND ----------

# MAGIC %md
# MAGIC --- pruebas paralelismo

# COMMAND ----------

import numpy as np
 
from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
 
from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK, Trials
 
import mlflow

# COMMAND ----------

X, y = load_boston(return_X_y=True)

# COMMAND ----------

# Review the mean value of each column in the dataset. You can see that they vary by several orders of magnitude, from 1425 for block population to 1.1 for average number of bedrooms. 
X.mean(axis=0)



# COMMAND ----------

from sklearn.preprocessing import StandardScaler
 
scaler = StandardScaler()
X = scaler.fit_transform(X)

# COMMAND ----------

# After scaling, the mean value for each column is close to 0. 
X.mean(axis=0)

# COMMAND ----------

y_discrete = np.where(y < np.median(y), 0, 1)

# COMMAND ----------

def objective(params):
    classifier_type = params['type']
    del params['type']
    if classifier_type == 'svm':
        clf = SVC(**params)
    elif classifier_type == 'rf':
        clf = RandomForestClassifier(**params)
    elif classifier_type == 'logreg':
        clf = LogisticRegression(**params)
    else:
        return 0
    accuracy = cross_val_score(clf, X, y_discrete).mean()
    
    # Because fmin() tries to minimize the objective, this function must return the negative accuracy. 
    return {'loss': -accuracy, 'status': STATUS_OK}

# COMMAND ----------

search_space = hp.choice('classifier_type', [
    {
        'type': 'svm',
        'C': hp.lognormal('SVM_C', 0, 1.0),
        'kernel': hp.choice('kernel', ['linear', 'rbf'])
    },
    {
        'type': 'rf',
        'max_depth': hp.quniform('max_depth', 2, 5, 1),
        'criterion': hp.choice('criterion', ['gini', 'entropy'])
    },
    {
        'type': 'logreg',
        'C': hp.lognormal('LR_C', 0, 1.0),
        'solver': hp.choice('solver', ['liblinear', 'lbfgs'])
    },
])

# COMMAND ----------


algo=tpe.suggest


# COMMAND ----------


spark_trials = SparkTrials()

# COMMAND ----------


with mlflow.start_run():
  best_result = fmin(
    fn=objective, 
    space=search_space,
    algo=algo,
    max_evals=32,
    trials=spark_trials)
