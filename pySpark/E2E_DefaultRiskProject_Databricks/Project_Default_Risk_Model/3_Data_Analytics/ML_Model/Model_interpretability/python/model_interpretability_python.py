# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # ML_Model 
# MAGIC 
# MAGIC ## Model_interpretability
# MAGIC 
# MAGIC Las técnicas de interpretabilidad o comunmente conocidas como eXplainable AI (XAI) son técnicas post-modelado para tratar de dar una interpretación a los resultados de modelos no transparentes. 
# MAGIC 
# MAGIC <div>
# MAGIC <img src="files/images/trade_off.png" align="center" width="500"/>
# MAGIC </div>
# MAGIC 
# MAGIC Fuente: Barredo Arrieta, et. al. (2019). Explainable Artificial Intelligence (XAI): Concepts, Taxonomies, Opportunities and Challenges toward Responsible AI. Information Fusion. 58. 10.1016/j.inffus.2019.12.012. 
# MAGIC 
# MAGIC Hay múltiples librerías que tratan de implementar técnicas de explicabilidad (open source, propietarias de AZ, AWS, etc). Sin duda alguna la que más marca la diferencia es [shap](https://shap.readthedocs.io/en/latest/index.html#).
# MAGIC 
# MAGIC SHAP (SHapley Additive exPlanations) es un método de relevancia de features para explicar la predicción que hace un determinado modelo en base a un score acumulado de las features que influyen. Su base matemática es la teoría de juegos cooperativos (para conocer más sobre XAI: [link](https://christophm.github.io/interpretable-ml-book/)).
# MAGIC 
# MAGIC <div>
# MAGIC <img src="files/images/shap_header.png" align="center" width="600"/>
# MAGIC </div>
# MAGIC 
# MAGIC Calcular los valores de SHAP es un trabajo computacionalmente intenso. Databriks permite la paralelización de este trabajo en un cluster de Spark.
# MAGIC 
# MAGIC Veamos cómo podemos usar esta librería en Databricks con nuestros modelos.

# COMMAND ----------

# MAGIC %run ../../../../0_Includes/configuration

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 1.- Import datasets and saved models

# COMMAND ----------

# MAGIC %md
# MAGIC Importamos un dataset sobre el que ejecutar nuestro modelo e interpretar los resultados. En este caso vamos a usar el dataset de test.

# COMMAND ----------

dict_config = {
  'storage_account_name' : storage_account_name,
  'container_name' : container_name,
  'folder_path' : folder_path_OUT,
  'file_name' : '/test' 
}

# import and transform it to pandas (python)
pandas_df = read_data_from_delta(dict_config, 
                               columns_to_keep=[c for c in columns_to_keep if c!='ID_CODCONTR'], 
                               to_pandas=True)

# Define classification labels & features
labels = pandas_df['CD_CODSUBCO'] == '6' # write-off
features = pandas_df.drop(['CD_CODSUBCO'], axis=1)

# categorical and numerical variables
cat_vars = [c for c in features.columns if c != 'IM_SCORIN']
num_vars = ['IM_SCORIN']


# COMMAND ----------

# MAGIC %md
# MAGIC Vamos a importar uno de los modelos guardados en las sesiones de mlflow que hemos creado en otros notebooks. Hay que tener en cuanta que nuestro modelo en realidad es una pipeline formada de varias etapas:

# COMMAND ----------

# load a previously saved model
# load it as a sklearn model so we can extract all the info from it
run_id = '74f8771ea5c14644a6632e135849de27'
model_pipe = mlflow.sklearn.load_model(
  'runs:/{run_id}/model'.format(run_id=run_id)
)

# print the different steps in the model pipeline
pprint(model_pipe.steps)

# COMMAND ----------

# MAGIC %md
# MAGIC Podemos acceder a las diferentes etapas de la pipeline porque mlflow entiende scikit pipelines. Por ejemplo, lo primero que se hace es aplicar un OHE a las variables categóricas. ¿Cómo renombra el modelo estas variables? En la siguiente celda extraemos los nombres dados.

# COMMAND ----------

# get the names given to the features after applying OHE
# OHE is the first step in the pre-processing pipeline
ohe_names = model_pipe['preprocessing_pipeline'].transformers_[0][1].get_feature_names()
ohe_names

# COMMAND ----------

# MAGIC %md
# MAGIC Podemos ver que el OHE ha dado nombres a las variables categóricas x_0, x_1, etc. La correspondencia con los nombres originales es la siguiente:

# COMMAND ----------

# dicctionary ohe name - category name
for i, name in enumerate(cat_vars):
  print(i, name)

# COMMAND ----------

# MAGIC %md
# MAGIC Ahora extraemos nuestro algoritmo ajustado, que es el último paso de la pipeline y dibujamos un gráfico de importancia de variables.

# COMMAND ----------

# the last step (the RF itself) contains the fitted model
model = model_pipe['model']

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.- Importancia de variables según el modelo

# COMMAND ----------

# RF models provides feature importance calculated as an average over the ensemble
importances = model.feature_importances_
feature_names = list(ohe_names) + list(num_vars)
feat_imp = pd.DataFrame({'feature':feature_names, 'importance':importances})

#top n 
top_n = 10
feat_imp = feat_imp.iloc[:top_n]
feat_imp.sort_values(by='importance', ascending=False, inplace=True)
feat_imp = feat_imp.set_index('feature', drop=True)
feat_imp.plot.barh(title='RF feature importance')
plt.xlabel('Feature Importance Score')
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 2.- SHAP Summary Plot

# COMMAND ----------

# explain the first 10 samples of the test

X_test = features[:10]

# COMMAND ----------

## compute SHAP values

# define the explainer
explainer = shap.TreeExplainer(model)

# first get the preprocessing transformer
preprocesor = model_pipe['preprocessing_pipeline']

# apply the transformer to observations
observations = preprocesor.transform(X_test).toarray()

# get Shap values from preprocessed data
shap_values = explainer.shap_values(observations)

# COMMAND ----------

len(shap_values)

# COMMAND ----------

labels.unique()

# COMMAND ----------

class_names = labels.unique()

shap.summary_plot(shap_values, observations, plot_type="bar"
                  , class_names=class_names, feature_names=feature_names
                 )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.- SHAP Force plot
# MAGIC 
# MAGIC Los graficos de fuerza dan explicabilidad de una predicción específica. Es un gráfico en el que se puede ver cómo las distintas variables contribuyen a la predicción que da el modelo. Es muy útil para hacer análisis de error o tener un conocimiento más profundo de un determinado caso particular.

# COMMAND ----------

# set a single observation
i=8

shap.force_plot(explainer.expected_value[0],
                shap_values[0][i], 
                observations[i], 
                feature_names=feature_names,
                matplotlib=True)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 4.- SHAP waterfall plot
# MAGIC 
# MAGIC El gráfico de cascada también proporciona explicabilidad local similar al de fuerza.

# COMMAND ----------

# set a single observation
i=8

shap.waterfall_plot(
  
  shap.Explanation(values=shap_values[0][i], 
                   base_values=explainer.expected_value[0], 
                   data=observations[i], 
                   feature_names=feature_names)
)
