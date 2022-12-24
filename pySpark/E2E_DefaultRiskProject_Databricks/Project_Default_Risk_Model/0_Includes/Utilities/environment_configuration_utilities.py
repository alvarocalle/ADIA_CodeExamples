# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC #### Environment configuration utilities
# MAGIC 
# MAGIC - Notebook used to define all global variables, libraries, data path, hive metastore objects, hive metastore permissions. 
# MAGIC 
# MAGIC - This notebook may use the `hive_metastore_utilities` and `data_path_utilities` so needs to be run afterwards.
# MAGIC 
# MAGIC     + Hive_metastore_utilities: contains support functions for hive metastore, the objects definition databases, tables, views, and security roles.  If the environment has a high number of objects this can use independent notebooks, json or external metadata store for the object’s definitions.
# MAGIC     + Data_path_utilities: contains support functions for the external storage containers path.

# COMMAND ----------

# DBTITLE 1,Global variables
RANDOM_SEED = 49

# COMMAND ----------

# DBTITLE 1,Imports
# spark io libraries
import pyspark.sql.functions as F
import pyspark.sql.types as T

# spark ml (be carefull with name convention conflicts)
from pyspark.ml.feature import RFormula
from pyspark.ml.feature import OneHotEncoder as mlOneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.feature import StandardScaler as mlStandardScaler 
from pyspark.ml.classification import LogisticRegression as mlLogisticRegression
from pyspark.ml.classification import RandomForestClassifier as mlRandomForestClassifier
from pyspark.ml.classification import GBTClassifier
from pyspark.ml import Pipeline as mlPipeline
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, TrainValidationSplit

# core python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint
import pickle

# display all the columns and rows 
pd.set_option('display.max_rows', 300)
pd.set_option('display.max_colwidth', None)

# set figure size for single graphs 
#plt.rcParams['figure.figsize'] = [15, 6]

import warnings
warnings.filterwarnings('ignore')

# python ml
import mlflow
from mlflow.tracking import MlflowClient

from sklearn.pipeline import Pipeline as sklearnPipeline
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.linear_model import LogisticRegression # explicable
from sklearn.ensemble import RandomForestClassifier # no explicable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split

### python resampling para el balanceo de clases
from imblearn.pipeline import Pipeline as imbalancePipeline
from imblearn.over_sampling import SMOTE

### optimización de hiperperámetros
from hyperopt import fmin, tpe, hp, SparkTrials, Trials, STATUS_OK
from hyperopt.pyll import scope

## ptython interpretabilidad
import shap


# COMMAND ----------

# DBTITLE 1,Data path parameters
storage_account_name = '**********'
container_name = '**********'
folder_path_IN = '**********'
folder_path_OUT = '**********'
csv_folder = '/riskdata'
file_name = '/export_ITA2.dsv'

file_path_IN = f"abfss://{container_name}@{storage_account_name}.dfs.core.windows.net{folder_path_IN}"
file_path_OUT = f"abfss://{container_name}@{storage_account_name}.dfs.core.windows.net{folder_path_OUT}"

# COMMAND ----------

# DBTITLE 1,ML parameters
columns_to_keep = [
  'ID_CODCONTR',
  'CD_CODSUBCO',
  'CD_CODMIS',
  'CD_CODFINAL',
  'CD_CODTTTIT',
  'CD_IRFSSEGM',
  'CD_CODPROGR',
  'CD_BASPORTF',
  'CD_STECUSEG',
  'CD_CODBRAND',
  'CD_FINRESEG',
  'CD_CODPANAC',
  'CD_CODTIPER',
  'CD_CODPARES',
  'CD_CODCUSTSEG1',
  'CD_CODCCCBC',
  'CD_VARCUSTUTP',

  'FLG_INDTIPAR',
  'FLG_INDCARES',
  'FLG_INDFORCUS',
  'FLG_INDDPLCU',

  'IM_SCORIN'
]

target_column = 'CD_CODSUBCO'

# COMMAND ----------

# DBTITLE 1, Date/DateTime Variables
from datetime import datetime
from datetime import date
from pytz import timezone
today_date = date.today().strftime("%Y-%m-%d")
now_time = datetime.now(timezone('Europe/Madrid')).strftime("%Y-%m-%d %H:%M:%S")
