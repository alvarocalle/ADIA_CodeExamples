# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Exploratory Data Analysis
# MAGIC 
# MAGIC - Exploration of the raw dataset to understand the different preprocessing steps needed for modeling.

# COMMAND ----------

# MAGIC %run ../../0_Includes/configuration

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Importamos el dataset guardado como CSV

# COMMAND ----------

dict_config = {
  'storage_account_name' : storage_account_name,
  'container_name' : container_name,
  'folder_path' : folder_path_IN + csv_folder,
  'file_name' : file_name 
}

dict_config

# COMMAND ----------


df_raw = read_data_from_csv(dict_config)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Análisis exploratorio del dato (EDA)

# COMMAND ----------

# MAGIC %md
# MAGIC Lo primero que tenemos que hacer es explorar el dato de forma general, mirar cómo es nuestra variable target, mirar cuantas columnas contienen información útil, etc.

# COMMAND ----------

# variable target: posibles valores
df_raw.groupBy(F.col("CD_CODSUBCO")).count().show()

# COMMAND ----------

# nos quedamos solo con los contratos con CD_CODSUBCO = 1 y 6
# 1 Canceled due maturity
# 6 Written-Off
df_raw = df_raw.filter(F.col("CD_CODSUBCO").isin(1,6))

# COMMAND ----------

# class imbalance (0.96, 0.04)
df.groupBy('CD_CODSUBCO').count().orderBy('CD_CODSUBCO').show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.1. Profiling
# MAGIC 
# MAGIC Databricks proporciona una herramienta muy potente para hacer profiling de datos. Con el profiling podemos hacernos una idea rápida de cual es la proporción de nulos, cómo se distribuye el dato, la cantidad de nulos, distintos, etc. Es una herramienta similar al pandas-profiling, pero permite usar la potencia de Spark cuando la cantidad de dato es muy alta. Hacer el profiling de un dataset grande es costoso. Databricks lo hace parcialmente y extrapola el cálculo.
# MAGIC 
# MAGIC Para usar la función `profiling` de Databricks hay que hacer display del spark dataframe
# MAGIC 
# MAGIC ```
# MAGIC display(df)
# MAGIC ```
# MAGIC 
# MAGIC Paro antes de usar la función display, vamos a contar la proporción exacta de nulos y no nulos por columna. Para ello uso una función definida en `ML_UDFS_utilities`

# COMMAND ----------

# numero de filas en el dataset
df_raw.count()

# COMMAND ----------

# esta función devuelve un pandas dataframe con los resultados
# en realidad en lugar de devolver un pandas podríamos 
# persistir el resultado en un fichero y despues leerlo
result = nulls_count_dataframe(df_raw)
result

# COMMAND ----------

# podemos salvar el resultado a un fichero para analizarlo detenidamente más adelante
dict_config = {
  'storage_account_name' : storage_account_name,
  'container_name' : container_name,
  'folder_path' : folder_path_OUT,
  'file_name' : '/nulls.csv' 
}
save_pandas_to_csv(dict_config, result)

# COMMAND ----------

# columnas que tienen todos los valores nulos
all_null_cols = result.loc[result['notNulls'] == 0, 'index'].values
all_null_cols

# COMMAND ----------

# eliminamos esas columnas
df = df_raw.drop(*all_null_cols)

# COMMAND ----------

# crea profiling para estudiar el resto de columnas (herramienta de Databricks optimizada)
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC A traves del profiling vemos que hay ciertas columnas que no proporcionan información.
# MAGIC 
# MAGIC Lo primero que hacemos es eliminarlas.

# COMMAND ----------

# Further cleanssing attending profiling
to_drop = [
  # numericas segun prodiling de databricks
  'CD_CODENTID', # - todos 934
  'CD_CODENTOR', # - todo 0
  'CD_INDFACAD', # - 99% son 0
  'CD_CODIPLIB', # - todos son 4
  'FLG_FECULINR', # - todos 0
  'IM_REPEXRAT', # - todo 1
  'IM_CORREXRA', # - todo 1
  'FLG_INDGCB', # - todo 0
  'DS_CODGRUPO', # - 97% es faltante
  'FLG_INDSEGES', # - todo 0
  'CD_CODIDEISST', # - todo 100k
  'CD_VARCOMREPOR', # - todo 934
  ##**CD_PRODGROUSCF - 55% missing, el resto todo 1
  'CD_SUBCONTRPART', # - 96% es missing
  'NU_NUMEMPLE', # - 95% missing
  'CD_CODSIEPE', # - 99% es 0
  'CD_CODMOSCA_1', # - 99% missing
  'IM_NUMSCORI', # - 97% missing
  'FLG_INDFORCUS' # - 99% es 0
  'CD_CLIENTTYP', # - 88% missing, resto 1
  'CD_CUSTSTATUS', # - 88% missing, resto 1
  'ID_IDBDRCUST', #- 99% missing

  # categoricas segun prodiling de databricks
  'CD_UNIDAD', # - todo ITA
  'CD_CODPAORI', #  - todo ITA
  'CD_CODMONED', # - todo EUR
  'CD_MANCOUN', # - todo ITA
  'FLG_INDCANLIMIT', # - todo N
  'FLG_TRDFINFLAG', # - 55% missing, resto N
  'FLG_INDCAPTIVE', # - 55% missing, resto Y
  'FLG_FLGLEASINGOP', # -  55% missing, resto N
  'NO_NOMPERSO', # - 90% missing
  'CD_CODTAMEM', # > 95% missing
  'FE_FECULRRI', # > 95% missing
  'FE_CODLEIPE', # > 95% missing
  'FE_CODGRFEV', # > 95% missing
  'CD_CODCVACO', # > 95% missing
  'CD_INDINTRAG', # > 95% missing
  'FE_DATSTARTUTPCUS', # > 95% missing
  'FE_DATENDUTPCUS', # > 95% missing
  'IM_NUMFACT', # > 95% missing
  'IM_NUMCUSTASS', # > 95% missing
  
  # variables categoricas con una proporción de missing > 50%
  'CD_VARLOCUSTSEG1',
  'CD_VARACCSECT',
  'CD_CODGFEVE',
  'CD_PRODGROUSCF',
  'CD_CODSEGIR',
  'CD_CLIENTTYP',
  'CD_CLASSIFCOMP',
  'DS_DESLOCSEGTY1',

  # variables repetidas
  'FE_DATO_1',
  'CD_UNIDAD_1'
]

df = df.drop(*to_drop)

# COMMAND ----------

# MAGIC %md
# MAGIC Hemos reducido el dataset, hemos eliminado todas aquellas columnas que no proporcionan información. 
# MAGIC 
# MAGIC El dataset ahora es lo suficientemente pequeño para que podamos usar Pandas y librería de Python puras.
# MAGIC 
# MAGIC Podría darse el caso, por la naturaleza del problema o el dataset, de que necesitásemos hacer esto usando pySpark. Cuando los datasets son pequeños es más práctico usar python puro y trabajar en memoria.

# COMMAND ----------

dataset = df.toPandas()

# COMMAND ----------

dataset.shape

# COMMAND ----------

# eliminamos todos los registros duplicados
df_ = dataset.drop_duplicates()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.2. Análisis univariante

# COMMAND ----------

df_.columns


# COMMAND ----------

# tipo de variables
ids = [var for var in df_.columns if 'ID_' in var]
fechas = [var for var in df_.columns if 'FE_' in var]
codigos = [var for var in df_.columns if 'CD_' in var]

# COMMAND ----------

# todas estas variables son identificadores (de contrato, personas, sucursal, etc)
# Son variables que servirían para lincar con otra info de las tablas pero a nivel de predictivo no tienen valor
ids

# COMMAND ----------

# todas estas variables están relacionadas con fechas (de reporte, de scoring, etc)
# en principio por la naturaleza del modelo que queremos crear vamos a prescindir de ellas
fechas

# COMMAND ----------

# estas variables contienen información codificada en forma de variables categóricas
# son variables relevantes para nuestro modelo
codigos

# COMMAND ----------

# MAGIC %md
# MAGIC Antes de nada, vamos a eliminar contratos que han cambiado de estado

# COMMAND ----------

# contratos que han cambiado de estado 1 -> 6 o viceversa
# hay tres contratos que han cambiado de estado, para este análisis los vamos a eliminar
a = df_[['ID_CODCONTR', 'CD_CODSUBCO']].groupby('ID_CODCONTR').nunique().reset_index()
contract_with_change = a.loc[a['CD_CODSUBCO'] > 1, 'ID_CODCONTR'].tolist()
contract_with_change

# COMMAND ----------

df_ = df_[~df_['ID_CODCONTR'].isin(contract_with_change)]

# COMMAND ----------

# MAGIC %md
# MAGIC Vamos a conocer un poco mejor las variables que tenemos

# COMMAND ----------

# CD_CODSUBCO: Derecognition Reason Code
# variable target
# podemos apreciar una gran desbalance de clases que tendremos que tener en cuenta a la hora de construir el clasificador

fig = plt.figure()
df_['CD_CODSUBCO'].value_counts(normalize=True).mul(100).plot.bar()
plt.xticks(rotation= 0)
plt.xlabel("Proporción")
plt.show()

# COMMAND ----------

# CD_CODFINAL: Purpose for which the loan was requested, according to the basic portfolio.

# Propósitos presentes en el dataset:
# CA122 Durables
# CA111 Auto New
# CA112 Auto Used
# CA124	Other
# CA123 Direct
# CA113 Motorcycles
# CA215 STF
# CA115 Leisure & Other
# CA121 Cards
# CA114 Caravans & Motorhomes

# Propósitos NO presentes en el dataset:
#CA131	Mortgage
#CA141	Other
#CA211	Stock Finance Auto
#CA212	Leasing
#CA213	Renting
#CA214	LTF
#CA221	LTF
#CA222	STF
#CA231	Leasing
#CA232	Renting
#CA233	LTF
#CA234	STF

fig = plt.figure()
df_['CD_CODFINAL'].value_counts(normalize=True).mul(100).plot.bar()
plt.title('Loan purposes', fontsize=16)
plt.ylabel("%")
plt.show()

# COMMAND ----------

# CD_CODTTTIT: Type of product according to FINREP's breakdown
#1	On demand (call) and short notice (current account)
#2	Credit card debt
#3	Trade receivables
#4	Finance leases
#6	Other term loans

print('Proporciones')
print(df_['CD_CODTTTIT'].value_counts(normalize=True).mul(100))

fig = plt.figure()
df_['CD_CODTTTIT'].value_counts(normalize=True).mul(100).plot.bar()
plt.xticks(rotation= 0)
plt.title('Product', fontsize=16)
plt.ylabel("%")
plt.show()


# COMMAND ----------

# CD_CODMOSCA: Especifica si el modelo utilizado es Rating o Scoring

print('Proporciones')
print(df_['CD_CODMOSCA'].value_counts(normalize=True).mul(100))

fig = plt.figure()
df_['CD_CODMOSCA'].value_counts(normalize=True).mul(100).plot.bar()
plt.title('Rating or Scoring', fontsize=16)
plt.xticks(rotation= 0)
plt.ylabel("%")
plt.show()

# COMMAND ----------

# CD_CODMIS: Producto definido en el MIS 2.0.

print('Proporciones')
print(df_['CD_CODMIS'].value_counts(normalize=True).mul(100))

fig = plt.figure()
df_['CD_CODMIS'].value_counts(normalize=True).mul(100).plot.bar()
plt.title('Product MIS', fontsize=16)
plt.xticks(rotation= 90)
plt.ylabel("%")
plt.show()


# COMMAND ----------

# CD_IRFSSEGM: Segmento IFRS9
#1	Auto New
#2	Auto Used
#4	Card Revolving
#6	Durables
#7	Direct
#9	Stock Finance Auto
#10	Leasing
#12	CQS

print('Proporciones')
print(df_['CD_IRFSSEGM'].value_counts(normalize=True).mul(100))

fig = plt.figure()
df_['CD_IRFSSEGM'].value_counts(normalize=True).mul(100).plot.bar()
plt.title('IRFSSEGM', fontsize=16)
plt.xticks(rotation= 90)
plt.ylabel("%")
plt.show()


# COMMAND ----------

# CD_CODPROGR: Código de producto agrupado según baremo, asociado al contrato
#1	Credits
#2	Letters of Credit 
#6	Cards
#8	Leasing
#10	Factoring w Resources
#11	Factoring w/o Resources
#181	Goods Consumer Loans 
#182	New Autos Consumer Loans
#183	Used Autos Consumer Loans
#184	Other Consumer Loans
#185	Salary Loans

print('Proporciones')
print(df_['CD_CODPROGR'].value_counts(normalize=True).mul(100))

fig = plt.figure()
df_['CD_CODPROGR'].value_counts(normalize=True).mul(100).plot.bar()
plt.title('CODPROGR', fontsize=16)
plt.xticks(rotation= 0)
plt.ylabel("%")
plt.show()

# COMMAND ----------

# CD_STECUSEG Portfolio Segmentation according to the Quarterly STE - Credit & Concentration exercise.
#2	Large Corporate (Annual turnover or Assets>= €50M)
#8	Retail - Other SME
#9	Retail - Other non-SME

print('Proporciones')
print(df_['CD_STECUSEG'].value_counts(normalize=True).mul(100))

fig = plt.figure()
df_['CD_STECUSEG'].value_counts(normalize=True).mul(100).plot.bar()
plt.title('STECUSEG', fontsize=16)
plt.xticks(rotation= 0)
plt.ylabel("%")
plt.show()


# COMMAND ----------

# CD_CODCCCBC: Contract Commercial Channel Code (Global Encoder)
# 1	Branches
# 3	Agents
# 5	SCF Sites
# 9	Car dealer

fig = plt.figure()
df_['CD_CODCCCBC'].value_counts(normalize=True).mul(100).plot.bar()
plt.title('Channel', fontsize=16)
plt.xticks(rotation= 0)
plt.ylabel("%")
plt.show()


# COMMAND ----------

# CD_IRFSSEGM: Aggregation level (segment) established according to IFRS9
# 1	Auto New
# 2	Auto Used
# 4	Card Revolving
# 6	Durables
# 7	Direct
# 9	Stock Finance Auto
# 10 Leasing
# 12 CQS

fig = plt.figure()
df_['CD_IRFSSEGM'].value_counts(normalize=True).mul(100).plot.bar()
plt.title('IRFSSEGM', fontsize=16)
plt.xticks(rotation= 0)
plt.ylabel("%")
plt.show()


# COMMAND ----------

cols = [
  'CD_BASPORTF', # Segmentación de acuerdo a las carteras básicas definidas según el enfoque de negocio para uso en reporting.
  'CD_CODBRAND', # Identification code of the brand associated with the customer.
  'CD_FINRESEG', # Type of counterparty according to the EBA 's breakdown (clients)
  'CD_CODPANAC', # Country of nationality of the different Intervenients of the contract.
  #'CD_CODSEINS', # codigo de identificación local (es un ID, a eliminar!!)
  'CD_CODTIPER', # "Código de Tipo de Persona (Física o Jurídica).
  'CD_CODPARES', # Country of residence of the intervenients (client/intervenient and guarantors).
  'CD_CODCUSTSEG1', # "Identification code of the segment associated with the customer.
  #'CD_CODPROACTI', # Actividad productiva asociada al cliente (p. ej. CNAE) (> 99% desinformada, eliminar)
  #'CD_CODSIC', # Código de actividad productiva local asociada al cliente. (> 99% desinformada, eliminar)
  'CD_CODCCCBC', # "Channels are the means of interaction between a customer and a company.
  'CD_VARCUSTUTP' # Clientes marcados por el indicador de probable impago (UTP)
]

for var in cols:

  #print('Proporciones de la variable {}'.format(var))
  #print(df_[var].value_counts(normalize=True).mul(100))

  fig = plt.figure()
  df_[var].value_counts(normalize=True).mul(100).plot.bar()
  plt.title(var, fontsize=16)
  plt.xticks(rotation= 90)
  plt.ylabel("%")
  plt.show()


# COMMAND ----------

# resto de variables
other = [c for c in df_.columns if c not in ids and c not in fechas and c not in codigos]

# COMMAND ----------

other

# COMMAND ----------

fig, axs = plt.subplots(2, 2, figsize=(15, 10))

# FLG_INDTIPAR
# Identificador (si/no) que marca aquellas operaciones que forman parte de una cartera  titulizada. Se marcará con ‘Y’ o ‘N’ según corresponda
df_['FLG_INDTIPAR'].value_counts().plot.bar(ax=axs[0,0])
axs[0,0].set_title('FLG_INDTIPAR', fontsize=12)
axs[0,0].set_ylabel('Frequency')


# FLG_INDCARES
df_['FLG_INDCARES'].value_counts().plot.bar(ax=axs[0,1])
axs[0,1].set_title('FLG_INDCARES', fontsize=12)
axs[0,1].set_ylabel('Frequency')


# FLG_INDFORCUS
df_['FLG_INDFORCUS'].value_counts().plot.bar(ax=axs[1,0])
axs[0,1].set_title('FLG_INDFORCUS', fontsize=12)
axs[0,1].set_ylabel('Frequency')


# FLG_INDDPLCU
df_['FLG_INDDPLCU'].value_counts().plot.bar(ax=axs[1,1])
axs[0,1].set_title('FLG_INDDPLCU', fontsize=12)
axs[0,1].set_ylabel('Frequency')

plt.show()


# COMMAND ----------

# DS_DESBRAND: equivalente al código de la marca pero con descripción, a eliminar
# DS_CODGRODE: nombre del dealer asociado al contrato, a eliminar 
# DS_DESLOCSEGTY1: Identification code of the segment associated with the customer (> 50% nulos)


# COMMAND ----------

# MAGIC %md
# MAGIC Variables numéricas

# COMMAND ----------

# IM_SCORIN Scores obtained from the statistic model (if any) prior to calibration for every intervinient (debtor and guarantors).
# IM_INITSCOR Original scoring obtained from the internal model set to the contract. 

fig, axs = plt.subplots(1, 2, figsize=(12, 4))

# distribution scoring
# de acuerdo a la descripción este scoring va de 0 a 1000 puntos, 
# sin embargo vemos que tenemos una cierta proporción de scores negativos
df_['IM_SCORIN'].hist(bins=7,ax=axs[0])
axs[0].set_title('Distribution of IM_SCORIN', fontsize=16)
axs[0].set_ylabel("Frequency")
axs[0].set_xlabel("Score")

# distribution initial scoring
# el comportamiento de esta variable es idéntico a la anterior, parece redundante
df_['IM_INITSCOR'].hist(bins=7,ax=axs[1])
axs[1].set_title('Distribution of IM_INITSCOR', fontsize=16)
axs[1].set_ylabel("Frequency")
axs[1].set_xlabel("Score")

plt.show()

# COMMAND ----------

df_[df_['IM_SCORIN'] < 0].shape[0]/len(df_)*100

# COMMAND ----------

# limpiamos la variable scoring
df_ = df_[df_['IM_SCORIN'] >= 0]


# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.2. Análisis multivariante

# COMMAND ----------

# 
cols = [
  'CD_BASPORTF', # Segmentación de acuerdo a las carteras básicas definidas según el enfoque de negocio para uso en reporting.
  'CD_CODBRAND', # Identification code of the brand associated with the customer.
  'CD_FINRESEG', # Type of counterparty according to the EBA 's breakdown (clients)
  'CD_CODPANAC', # Country of nationality of the different Intervenients of the contract.
  #'CD_CODSEINS', # codigo de identificación local (es un ID, a eliminar!!)
  'CD_CODTIPER', # "Código de Tipo de Persona (Física o Jurídica).
  'CD_CODPARES', # Country of residence of the intervenients (client/intervenient and guarantors).
  'CD_CODCUSTSEG1', # "Identification code of the segment associated with the customer.
  #'CD_CODPROACTI', # Actividad productiva asociada al cliente (p. ej. CNAE) (> 99% desinformada, eliminar)
  #'CD_CODSIC', # Código de actividad productiva local asociada al cliente. (> 99% desinformada, eliminar)
  'CD_CODCCCBC', # "Channels are the means of interaction between a customer and a company.
  'CD_VARCUSTUTP' # Clientes marcados por el indicador de probable impago (UTP)
]

for col in cols:
  pd.crosstab(df_['CD_CODSUBCO'], df_[col], normalize="index").mul(100).plot(kind='bar', figsize=(20,5))
  plt.title('Loan status by {}'.format(col), fontsize=16)
  plt.xticks(rotation = 0)
  plt.ylabel('%')
  plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Tratamiento de nulos

# COMMAND ----------

selected_variables = [
  'CD_CODSUBCO',
  'CD_CODMIS',
  'CD_CODMOSCA',
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

# COMMAND ----------

percent_missing = df_[selected_variables].isnull().sum() * 100 / len(df_)
missing_value_df = pd.DataFrame({'column_name': df_[selected_variables].columns,
                                 'percent_missing': percent_missing})\
                      .sort_values('percent_missing')
missing_value_df

# COMMAND ----------

# MAGIC %md
# MAGIC Las variables con contenido en nulos son todas categóricas. 
# MAGIC 
# MAGIC Aqui vamos a usar una imputación bastante común cuando se trata de este tipo de variables. Vamos a crear una nueva categoría que va a indicar que tenemos registros faltantes 'None'.

# COMMAND ----------

df_ = df_[selected_variables].fillna('None')

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4. Encoding

# COMMAND ----------

# MAGIC %md
# MAGIC We will perform OHE for the categorical variables.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5. Conclusiones del análisis
# MAGIC 
# MAGIC - Filtrar los contratos con CD_CODSUBCO = 1 y 6
# MAGIC 
# MAGIC - Crear una lista de variables a usar y leer solo esas columnas
# MAGIC 
# MAGIC - Conservamos la variable `IM_SCORIN` pero eliminamos registros con scoring negativo (0.2 %) y la normalizamos (0, 1000) -> (0, 1). Para ello usamos `sklearn - > preprocessing.StandardScaler()`
# MAGIC 
# MAGIC - Los faltantes se tratan creando una nueva categoría `None` ya que estamos trabajando con categóricas. 
# MAGIC 
# MAGIC - Encoding de variables
