# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC #### ML UDFS utilities
# MAGIC 
# MAGIC - Definition of user defined functions that support the ML process

# COMMAND ----------

# MAGIC %md
# MAGIC #### Exploration utilities

# COMMAND ----------

def nulls_count_column(coldata):
  """ function that recieves a pyspark dataframe column and returns 
      the number of nulls and not nulls as a tuple
  Arg:
    coldata : pyspark-dataframe column in the form of df.select(<name_of_column>)
  Returns:
    tuple : (number_of_nulls, number_of_not_nulls)
  """
  colname = coldata.columns[0]
  check_nulls = F.sum(F.when(F.isnan(colname)|F.col(colname).isNull(),F.lit(1)).otherwise(F.lit(0))).alias("nulls") if isinstance(coldata.schema.fields[0].dataType,T.NumericType) else F.sum(F.when(F.col(colname).isNull(),F.lit(1)).otherwise(F.lit(0))).alias("nulls")
  check_not_nulls = F.count(F.when(~F.isnan(colname) & F.col(colname).isNotNull(),F.col(colname))).alias("notNulls") if isinstance(coldata.schema.fields[0].dataType,T.NumericType) else F.count(F.when(F.col(colname).isNotNull(),F.col(colname))).alias("notNulls")
  num_nulls = coldata.select(check_nulls, check_not_nulls).first()
  
  return num_nulls["nulls"],num_nulls["notNulls"]

# COMMAND ----------

def nulls_count_dataframe(df):
  """ function that receives a pyspark dataframe and calculates the nulls/notnulls for each column
  Args:
    df : pyspark dataframe
  Returns:
    pandas : a pandas dataframe with the calculated number of nulls and not nulls per column
  Example:
  >>> result = nulls_count_dataframe(df)
  """
  
  #calculate nulls and notnulls per column:
  results = pd.DataFrame(index=df_raw.columns, columns=["nulls","notNulls"])
  for colname in df.columns:
    results.loc[colname, ["nulls","notNulls"]] = nulls_count_column(df_raw.select(colname))
  return results.reset_index()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Preprocessing utilities (specific for the problem at hands)

# COMMAND ----------

def cleanssing(df):
  """ function that makes a series of cleanssing steps in a pyspark dataframe 
      the steps have been deduced in the EDA phase
    Args:
        df (pyspark-like dataframe) : raw data
    Returns:
        pyspark dataframe : a clean dataframe
    Example:
    >>> df_clean = cleanssing(df_raw)
  """

  # filtra contratos
  df = df.filter(F.col("CD_CODSUBCO").isin(1,6))
  
  # eliminar registros duplicados
  df = df.drop_duplicates()

  # contratos que han cambiado de estado 1 -> 6 o viceversa
  a = df.select('ID_CODCONTR', 'CD_CODSUBCO')\
        .groupBy('ID_CODCONTR').agg(F.countDistinct('CD_CODSUBCO').alias('distinctStates'))\
        .filter(F.col('distinctStates') > 1)\
        .select('ID_CODCONTR').rdd.flatMap(lambda x: x).collect()

  df = df.filter(F.col('ID_CODCONTR').isin(a) == False)\
          .drop(F.col('ID_CODCONTR'))

  # limpiamos la variable scoring
  df = df.filter(F.col('IM_SCORIN') >= 0)

  # tratamiento de missing
  # para poder eliminarlos lo transformo todo a string y uso la función na.fill()
  for c in df.columns:
    df = df.withColumn(c, F.col(c).cast(T. StringType()))
  df = df.na.fill('unknown') 
  
  return df

# COMMAND ----------

def sparkDF_stratified_train_test_split(df, target_column, labels, train_fraction=0.8, random_state=1):
  """ function that does a train/test split of a pyspark dataframe for classification problems 
  Args:
    df (pyspark dataframe) : original dataset
    target_column (str) : target column name
    labels (list) : list with labels
    train_fraction (float) : fraction of train set (a number in 0-1)
    random_state (int) : random seed for reproducibility
  Returns:
    train, test : two pyspark dataframes
  Example:
  >>> labels=[1, 6]
  >>> train_fraction=0.8
  >>> target_column='CD_CODSUBCO'
  >>> train, test = sparkDF_stratified_train_test_split(dataset, target_column, labels, 0.8, 49)
  """
  train = df.sampleBy(target_column, fractions={label:train_fraction for label in labels}, seed=random_state)
  test = df.subtract(train)
  return train, test

# COMMAND ----------

# MAGIC %md
# MAGIC #### Modelling utilities
# MAGIC 
# MAGIC   

# COMMAND ----------

def preprocessing(X, categorical_columns, numerical_columns):
  """ function to perform preprocessing steps (ohe & scaling) in features
  Args:
    X (pandas-like) : feature values, shape (n_rows, n_features)
    categorical_columns (str): list of categorical column names
    numerical_columns (str): list of numerical column names
  Returns:
    X_processed (numpy-array) : matrix with processed features
  Example:  
  >>> x_transformed = preprocessing(X, categorical_columns, numerical_columns)
  """
  
  categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")
  numerical_preprocessor = StandardScaler()

  preprocessor = ColumnTransformer([
      ('one-hot-encoder', categorical_preprocessor, categorical_columns),
      ('standard_scaler', numerical_preprocessor, numerical_columns)
      ])
  
  return preprocessor.fit(features)

# COMMAND ----------

def model_training(estimator, X, y, categorical_columns, numerical_columns, balance=False):
  """ function to train a scikit-type classifier to a supervised dataset (X, y)
      the function takes care of performing ohe, scaling and SMOTE sampling previous to training  
  Args:
    estimator (scikit-like classifier) : a classifier
    X (pandas-like) : feature values, shape (n_rows, n_features)
    y (pandas-like) : binary target values (0/1 or True/False), shape (n_rows, 1)
    categorical_columns (str): list of categorical column names
    numerical_columns (str): list of numerical column names
  Returns:
    scikit-type pipeline : a fitted pipeline to (X, y) with steps ('preprocessing_pipeline, 'sampling', 'model')
  Example:  
  >>> estimator = LogisticRegression()
  >>> model_pipe = model_training(estimator, X_train, y_train, cat_vars, num_vars)
  >>> predictions = model_pipe.predict(X_test)
  """
  
  categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")
  numerical_preprocessor = StandardScaler()

  model_preprocessor = ColumnTransformer([
      ('one-hot-encoder', categorical_preprocessor, categorical_columns),
      ('standard_scaler', numerical_preprocessor, numerical_columns)
      ])
  
  if balance:
    model_pipeline = imbalancePipeline(
        steps = [
            ('preprocessing_pipeline', model_preprocessor),
            ('sampling', SMOTE()),
            ('model', estimator),
        ])
  else:
    model_pipeline = sklearnPipeline(
        steps = [
            ('preprocessing_pipeline', model_preprocessor),
            ('model', estimator),
        ])
  
  model_pipeline.fit(X, y)

  return model_pipeline

# COMMAND ----------

def fetch_logged_data(run_id):
  """ function to fetch metadata from a model saved with mlflow
  Args: 
    run_id (str) : model ID as saved by mlflow
  Returns:
    params (dict) : set of hyperparameters of the saved model
    metrics (dict) ; saved metrics
    tags (dict) : saved tags
    artifacts (list) : saved artifacts like conda.yaml, model.pkl, requirements.txt
  Example:
  >>> # manually specify the run id
  >>> run_id = 'd2ee757feb8f48e986b9a01d29003748'
  >>> # or also programatically, i.e., last run
  >>> run_id = run.info.run_id
  >>> # fetch logged data
  >>> params, metrics, tags, artifacts = fetch_logged_data(run_id)
  """
  
  client = mlflow.tracking.MlflowClient()
  data = client.get_run(run_id).data
  tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}
  artifacts = [f.path for f in client.list_artifacts(run_id, "model")]
  return data.params, data.metrics, tags, artifacts

# COMMAND ----------

def train_model_with_hpo(params):
  """
  This function builds the loss function for HPO. Here the loss function is -1*auc_score
  At the end an optimization problem is nothing but finding the minimum of a function 
  We want the auc_score to be the maximum in this case, se we need to minimize -1*auc_score
  """ 
  with mlflow.start_run(nested=True):
    
    # notice that now our estimator 
    estimator = RandomForestClassifier(
      random_state=0,
      **params
    )
    
    # this function automatically fits the model but we could do it outside
    model_pipe = model_training(estimator, X_train, y_train, cat_vars, num_vars)
    predicted_probs = model_pipe.predict_proba(X_test)
    
    # Tune based on the test AUC
    # In production settings, you could use a separate validation set instead
    roc_auc = roc_auc_score(y_test, predicted_probs[:,1])
    mlflow.log_metric('test_auc', roc_auc)
    
    # Set the loss to -1*auc_score so fmin maximizes the auc_score
    return {'status': STATUS_OK, 'loss': -1*roc_auc}


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Evaluation utilities

# COMMAND ----------

def eval_metrics(actual, pred):
  """  function to calculate classification metrics precision, recall and roc_auc
  Args:
    actual (numpy-array) : array of y_test values
    pred (numpy-array) : array of y_predict from model
  Returns:
    list : (precision, recall, roc_auc)
  Example:  
  >>> predictions = model_pipe.predict(X_test)
  >>> (precision, recall, roc_auc) = eval_metrics(y_test, predictions)
  """
  precision = precision_score(actual, pred)
  recall = recall_score(actual, pred)
  roc_auc = roc_auc_score(actual, pred)
  
  return precision, recall, roc_auc

# COMMAND ----------

def plot_roc(y_true, y_probas, title='ROC Curves', file_name = None,
                   plot_micro=True, plot_macro=True, classes_to_plot=None,
                   ax=None, figsize=None, cmap='nipy_spectral', 
                   title_fontsize="large", text_fontsize="medium"):
    """Generates the ROC curves from labels and predicted scores/probabilities

    Args:
        y_true (array-like, shape (n_samples)):
            Ground truth (correct) target values.

        y_probas (array-like, shape (n_samples, n_classes)):
            Prediction probabilities for each class returned by a classifier.
            
        file_name:(string, optional): Name of the generated plot. Defaults to None

        title (string, optional): Title of the generated plot. Defaults to
            "ROC Curves".

        classes_to_plot (list-like, optional): Classes for which the ROC
            curve should be plotted. e.g. [0, 'cold']. If given class does not exist,
            it will be ignored. If ``None``, all classes will be plotted. Defaults to
            ``None``

        ax (:class:`matplotlib.axes.Axes`, optional): The axes upon which to
            plot the curve. If None, the plot is drawn on a new set of axes.

        figsize (2-tuple, optional): Tuple denoting figure size of the plot
            e.g. (6, 6). Defaults to ``None``.

        cmap (string or :class:`matplotlib.colors.Colormap` instance, optional):
            Colormap used for plotting the projection. View Matplotlib Colormap
            documentation for available options.
            https://matplotlib.org/users/colormaps.html

        title_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values. Defaults to
            "large".

        text_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values. Defaults to
            "medium".

    Returns:
        ax (:class:`matplotlib.axes.Axes`): The axes on which the plot was
            drawn.

    Example:
        >>> import scikitplot as skplt
        >>> nb = GaussianNB()
        >>> nb = nb.fit(X_train, y_train)
        >>> y_probas = nb.predict_proba(X_test)
        >>> skplt.metrics.plot_roc(y_test, y_probas)
        <matplotlib.axes._subplots.AxesSubplot object at 0x7fe967d64490>
        >>> plt.show()

        .. image:: _static/examples/plot_roc_curve.png
           :align: center
           :alt: ROC Curves
    """
    y_true = np.array(y_true)
    y_probas = np.array(y_probas)

    classes = np.unique(y_true)
    probas = y_probas

    if classes_to_plot is None:
        classes_to_plot = classes

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.set_title(title, fontsize=title_fontsize)

    fpr_dict = dict()
    tpr_dict = dict()

    indices_to_plot = np.in1d(classes, classes_to_plot)
    for i, to_plot in enumerate(indices_to_plot):
        fpr_dict[i], tpr_dict[i], _ = metrics.roc_curve(y_true, probas[:, i],
                                                pos_label=classes[i])
        if to_plot:
            roc_auc = metrics.auc(fpr_dict[i], tpr_dict[i])
            color = plt.cm.get_cmap(cmap)(float(i) / len(classes))
            ax.plot(fpr_dict[i], tpr_dict[i], lw=2, color=color,
                  label='ROC curve of class {0} (area = {1:0.2f})'
                        ''.format(classes[i], roc_auc))


    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=text_fontsize)
    ax.set_ylabel('True Positive Rate', fontsize=text_fontsize)
    ax.tick_params(labelsize=text_fontsize)
    ax.legend(loc='lower right', fontsize=text_fontsize)
    ax.grid('on')
  
    fig.savefig(file_name)
    
    return ax

# COMMAND ----------

def cumulative_gain_curve(y_true, y_score, pos_label=None):
    """This function generates the points necessary to plot the Cumulative Gain

    Note: This implementation is restricted to the binary classification task.

    Args:
        y_true (array-like, shape (n_samples)): True labels of the data.

        y_score (array-like, shape (n_samples)): Target scores, can either be
            probability estimates of the positive class, confidence values, or
            non-thresholded measure of decisions (as returned by
            decision_function on some classifiers).

        pos_label (int or str, default=None): Label considered as positive and
            others are considered negative

    Returns:
        percentages (numpy.ndarray): An array containing the X-axis values for
            plotting the Cumulative Gains chart.

        gains (numpy.ndarray): An array containing the Y-axis values for one
            curve of the Cumulative Gains chart.

    Raises:
        ValueError: If `y_true` is not composed of 2 classes. The Cumulative
            Gain Chart is only relevant in binary classification.
    """
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)

    # ensure binary classification if pos_label is not specified
    classes = np.unique(y_true)
    if (pos_label is None and
        not (np.array_equal(classes, [0, 1]) or
             np.array_equal(classes, [-1, 1]) or
             np.array_equal(classes, [0]) or
             np.array_equal(classes, [-1]) or
             np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    sorted_indices = np.argsort(y_score)[::-1]
    y_true = y_true[sorted_indices]
    gains = np.cumsum(y_true)

    percentages = np.arange(start=1, stop=len(y_true) + 1)

    gains = gains / float(np.sum(y_true))
    percentages = percentages / float(len(y_true))

    gains = np.insert(gains, 0, [0])
    percentages = np.insert(percentages, 0, [0])

    return percentages, gains


# COMMAND ----------

def plot_cumulative_gain(y_true, y_probas,  file_name = None, curve_name = None,
                         title='Cumulative Gains Curve', 
                         ax=None, figsize=None, title_fontsize="large",
                         text_fontsize="medium"):
    """Generates the Cumulative Gains Plot from labels and scores/probabilities

    The cumulative gains chart is used to determine the effectiveness of a
    binary classifier. A detailed explanation can be found at
    http://mlwiki.org/index.php/Cumulative_Gain_Chart. The implementation
    here works only for binary classification.

    Args:
        y_true (array-like, shape (n_samples)):
            Ground truth (correct) target values.

        y_probas (array-like, shape (n_samples, n_classes)):
            Prediction probabilities for each class returned by a classifier.
            
        file_name:(string, optional): Name of the generated plot. Defaults to None
        
        file_name:(string, optional): Name of the curve plot. Defaults to None

        title (string, optional): Title of the generated plot. Defaults to
            "Cumulative Gains Curve".

        ax (:class:`matplotlib.axes.Axes`, optional): The axes upon which to
            plot the learning curve. If None, the plot is drawn on a new set of
            axes.

        figsize (2-tuple, optional): Tuple denoting figure size of the plot
            e.g. (6, 6). Defaults to ``None``.

        title_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values. Defaults to
            "large".

        text_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values. Defaults to
            "medium".

    Returns:
        ax (:class:`matplotlib.axes.Axes`): The axes on which the plot was
            drawn.
        roc_auc_correct (`PCT`): AUC of Gain Chart

    Example:
        >>> import scikitplot as skplt
        >>> lr = LogisticRegression()
        >>> lr = lr.fit(X_train, y_train)
        >>> y_probas = lr.predict_proba(X_test)
        >>> skplt.metrics.plot_cumulative_gain(y_test, y_probas)
        <matplotlib.axes._subplots.AxesSubplot object at 0x7fe967d64490>
        >>> plt.show()

        .. image:: _static/examples/plot_cumulative_gain.png
           :align: center
           :alt: Cumulative Gains Plot
    """
    y_true = np.array(y_true)
    y_probas = np.array(y_probas)

    classes = np.unique(y_true)
    if len(classes) != 2:
        raise ValueError('Cannot calculate Cumulative Gains for data with '
                         '{} category/ies'.format(len(classes)))

    # Compute Cumulative Gain Curves
    percentages, gains1 = cumulative_gain_curve(y_true, y_probas[:, 0],
                                                classes[0])
    percentages, gains2 = cumulative_gain_curve(y_true, y_probas[:, 1],
                                                classes[1])
    
    prior=y_true.mean()
    
    roc_auc = metrics.auc(percentages,gains2)
    roc_auc_correct = roc_auc/(1-prior/2)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.plot([0, prior, 1], [0, 1, 1], 'r', lw=2, label='Perfect Model')
        #ax.plot(percentages, gains2, lw=3, label='Class {}'.format(classes[0]))
        
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Baseline')

    ax.set_title(title, fontsize=title_fontsize)
    
    if curve_name == None:
        label_plot = 'Class {0} (area = {1:0.2f})'.format(classes[1], roc_auc_correct)
    else:
        label_plot = '{0} (area = {1:0.2f})'.format(curve_name, roc_auc_correct)        
         
    ax.plot(percentages, gains2, lw=3, label=label_plot)
        

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])


    ax.set_xlabel('Percentage of sample', fontsize=text_fontsize)
    ax.set_ylabel('Gain', fontsize=text_fontsize)
    ax.tick_params(labelsize=text_fontsize)
    ax.grid('on')
    ax.legend(loc='lower right', fontsize=text_fontsize)
    
    if file_name != None:
        fig.savefig(file_name)
    
    return ax, roc_auc_correct


# COMMAND ----------

def plot_precision_recall(y_true, y_probas, file_name,
                          title='Precision-Recall Curve',
                          plot_micro=False,
                          classes_to_plot=None, ax=None,
                          figsize=None, cmap='nipy_spectral',
                          title_fontsize="large",
                          text_fontsize="medium"):
    """Generates the Precision Recall Curve from labels and probabilities

    Args:
        y_true (array-like, shape (n_samples)):
            Ground truth (correct) target values.

        y_probas (array-like, shape (n_samples, n_classes)):
            Prediction probabilities for each class returned by a classifier.

        file_name:(string, optional): Name of the generated plot. Defaults to None
        
        title (string, optional): Title of the generated plot. Defaults to
            "Precision-Recall curve".

        plot_micro (boolean, optional): Plot the micro average ROC curve.
            Defaults to ``True``.

        classes_to_plot (list-like, optional): Classes for which the precision-recall
            curve should be plotted. e.g. [0, 'cold']. If given class does not exist,
            it will be ignored. If ``None``, all classes will be plotted. Defaults to
            ``None``.

        ax (:class:`matplotlib.axes.Axes`, optional): The axes upon which to
            plot the curve. If None, the plot is drawn on a new set of axes.

        figsize (2-tuple, optional): Tuple denoting figure size of the plot
            e.g. (6, 6). Defaults to ``None``.

        cmap (string or :class:`matplotlib.colors.Colormap` instance, optional):
            Colormap used for plotting the projection. View Matplotlib Colormap
            documentation for available options.
            https://matplotlib.org/users/colormaps.html

        title_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values. Defaults to
            "large".

        text_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values. Defaults to
            "medium".

    Returns:
        ax (:class:`matplotlib.axes.Axes`): The axes on which the plot was
            drawn.

    Example:
        >>> import scikitplot as skplt
        >>> nb = GaussianNB()
        >>> nb.fit(X_train, y_train)
        >>> y_probas = nb.predict_proba(X_test)
        >>> skplt.metrics.plot_precision_recall(y_test, y_probas)
        <matplotlib.axes._subplots.AxesSubplot object at 0x7fe967d64490>
        >>> plt.show()

        .. image:: _static/examples/plot_precision_recall_curve.png
           :align: center
           :alt: Precision Recall Curve
    """
    y_true = np.array(y_true)
    y_probas = np.array(y_probas)

    classes = np.unique(y_true)
    probas = y_probas

    if classes_to_plot is None:
        classes_to_plot = classes

    binarized_y_true = label_binarize(y_true, classes=classes)
    if len(classes) == 2:
        binarized_y_true = np.hstack(
            (1 - binarized_y_true, binarized_y_true))

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.set_title(title, fontsize=title_fontsize)

    indices_to_plot = np.in1d(classes, classes_to_plot)
    for i, to_plot in enumerate(indices_to_plot):
        if to_plot:
            average_precision = metrics.average_precision_score(
                binarized_y_true[:, i],
                probas[:, i])
            precision, recall, _ = metrics.precision_recall_curve(
                y_true, probas[:, i], pos_label=classes[i])
            color = plt.cm.get_cmap(cmap)(float(i) / len(classes))
            ax.plot(recall, precision, lw=2,
                    label='Precision-recall curve of class {0} '
                          '(area = {1:0.3f})'.format(classes[i],
                                                     average_precision),
                    color=color)

    if plot_micro:
        precision, recall, _ = precision_recall_curve(
            binarized_y_true.ravel(), probas.ravel())
        average_precision = average_precision_score(binarized_y_true,
                                                    probas,
                                                    average='micro')
        ax.plot(recall, precision,
                label='micro-average Precision-recall curve '
                      '(area = {0:0.3f})'.format(average_precision),
                color='navy', linestyle=':', linewidth=4)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.tick_params(labelsize=text_fontsize)
    ax.legend(loc='best', fontsize=text_fontsize)
    ax.grid('on')
    
    
    fig.savefig(file_name)
    return ax
  

# COMMAND ----------

def binary_ks_curve(y_true, y_probas):
    """This function generates the points necessary to calculate the KS
    Statistic curve.

    Args:
        y_true (array-like, shape (n_samples)): True labels of the data.

        y_probas (array-like, shape (n_samples)): Probability predictions of
            the positive class.

    Returns:
        thresholds (numpy.ndarray): An array containing the X-axis values for
            plotting the KS Statistic plot.

        pct1 (numpy.ndarray): An array containing the Y-axis values for one
            curve of the KS Statistic plot.

        pct2 (numpy.ndarray): An array containing the Y-axis values for one
            curve of the KS Statistic plot.

        ks_statistic (float): The KS Statistic, or the maximum vertical
            distance between the two curves.

        max_distance_at (float): The X-axis value at which the maximum vertical
            distance between the two curves is seen.

        classes (np.ndarray, shape (2)): An array containing the labels of the
            two classes making up `y_true`.

    Raises:
        ValueError: If `y_true` is not composed of 2 classes. The KS Statistic
            is only relevant in binary classification.
    """
    y_true, y_probas = np.asarray(y_true), np.asarray(y_probas)
    lb = LabelEncoder()
    encoded_labels = lb.fit_transform(y_true)
    if len(lb.classes_) != 2:
        raise ValueError('Cannot calculate KS statistic for data with '
                         '{} category/ies'.format(len(lb.classes_)))
    idx = encoded_labels == 0
    data1 = np.sort(y_probas[idx])
    data2 = np.sort(y_probas[np.logical_not(idx)])

    ctr1, ctr2 = 0, 0
    thresholds, pct1, pct2 = [], [], []
    while ctr1 < len(data1) or ctr2 < len(data2):

        # Check if data1 has no more elements
        if ctr1 >= len(data1):
            current = data2[ctr2]
            while ctr2 < len(data2) and current == data2[ctr2]:
                ctr2 += 1

        # Check if data2 has no more elements
        elif ctr2 >= len(data2):
            current = data1[ctr1]
            while ctr1 < len(data1) and current == data1[ctr1]:
                ctr1 += 1

        else:
            if data1[ctr1] > data2[ctr2]:
                current = data2[ctr2]
                while ctr2 < len(data2) and current == data2[ctr2]:
                    ctr2 += 1

            elif data1[ctr1] < data2[ctr2]:
                current = data1[ctr1]
                while ctr1 < len(data1) and current == data1[ctr1]:
                    ctr1 += 1

            else:
                current = data2[ctr2]
                while ctr2 < len(data2) and current == data2[ctr2]:
                    ctr2 += 1
                while ctr1 < len(data1) and current == data1[ctr1]:
                    ctr1 += 1

        thresholds.append(current)
        pct1.append(ctr1)
        pct2.append(ctr2)

    thresholds = np.asarray(thresholds)
    pct1 = np.asarray(pct1) / float(len(data1))
    pct2 = np.asarray(pct2) / float(len(data2))

    if thresholds[0] != 0:
        thresholds = np.insert(thresholds, 0, [0.0])
        pct1 = np.insert(pct1, 0, [0.0])
        pct2 = np.insert(pct2, 0, [0.0])
    if thresholds[-1] != 1:
        thresholds = np.append(thresholds, [1.0])
        pct1 = np.append(pct1, [1.0])
        pct2 = np.append(pct2, [1.0])

    differences = pct1 - pct2
    ks_statistic, max_distance_at = (np.max(differences),
                                     thresholds[np.argmax(differences)])

    return thresholds, pct1, pct2, ks_statistic, max_distance_at, lb.classes_


# COMMAND ----------

def plot_ks_statistic(y_true, y_probas, title='KS Statistic Plot',
                      ax=None, figsize=None, title_fontsize="large",
                      text_fontsize="medium",file_name=None):
    """Generates the KS Statistic plot from labels and scores/probabilities
    Args:
        y_true (array-like, shape (n_samples)):
            Ground truth (correct) target values.
        y_probas (array-like, shape (n_samples, n_classes)):
            Prediction probabilities for each class returned by a classifier.
        title (string, optional): Title of the generated plot. Defaults to
            "KS Statistic Plot".
        ax (:class:`matplotlib.axes.Axes`, optional): The axes upon which to
            plot the learning curve. If None, the plot is drawn on a new set of
            axes.
        figsize (2-tuple, optional): Tuple denoting figure size of the plot
            e.g. (6, 6). Defaults to ``None``.
        title_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values. Defaults to
            "large".
        text_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values. Defaults to
            "medium".
    Returns:
        ax (:class:`matplotlib.axes.Axes`): The axes on which the plot was
            drawn.
    Example:
        >>> import scikitplot.plotters as skplt
        >>> lr = LogisticRegression()
        >>> lr = lr.fit(X_train, y_train)
        >>> y_probas = lr.predict_proba(X_test)
        >>> skplt.plot_ks_statistic(y_test, y_probas)
        <matplotlib.axes._subplots.AxesSubplot object at 0x7fe967d64490>
        >>> plt.show()
        .. image:: _static/examples/plot_ks_statistic.png
           :align: center
           :alt: KS Statistic
    """
    y_true = np.array(y_true)
    y_probas = np.array(y_probas)

    classes = np.unique(y_true)
    if len(classes) != 2:
        raise ValueError('Cannot calculate KS statistic for data with '
                         '{} category/ies'.format(len(classes)))
    probas = y_probas

    # Compute KS Statistic curves
    thresholds, pct1, pct2, ks_statistic, \
        max_distance_at, classes = binary_ks_curve(y_true,
                                                   probas[:, 1].ravel())

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.set_title(title, fontsize=title_fontsize)

    ax.plot(thresholds, pct1, lw=3, label='Class {}'.format(classes[0]))
    ax.plot(thresholds, pct2, lw=3, label='Class {}'.format(classes[1]))
    idx = np.where(thresholds == max_distance_at)[0][0]
    ax.axvline(max_distance_at, *sorted([pct1[idx], pct2[idx]]),
               label='KS Statistic: {:.3f} at {:.3f}'.format(ks_statistic,
                                                             max_distance_at),
               linestyle=':', lw=3, color='black')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])

    ax.set_xlabel('Threshold', fontsize=text_fontsize)
    ax.set_ylabel('Percentage below threshold', fontsize=text_fontsize)
    ax.tick_params(labelsize=text_fontsize)
    ax.legend(loc='lower right', fontsize=text_fontsize)
    ax.grid('on')
    
    ks_size = (probas[:, 1].ravel()>max_distance_at).sum()/len(y_true)
    
    fig.savefig(file_name)
    
    return ax, ks_statistic, max_distance_at, ks_size
