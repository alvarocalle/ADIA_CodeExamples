#----------------------------------
#----------------------------------

import re
from pyspark.sql.types import DoubleType, StructType, StructField

"""
Binary classification metrics
"""
def accuracy(y_true, y_pred):
    """
    Function to calculate accuracy
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: accuracy score
    """
    correct_counter = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == yp:
            correct_counter += 1
    return correct_counter / len(y_true)

def true_positive(y_true, y_pred):
    """
    Function to calculate True Positives
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: number of true positives
    """
    tp = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 'X' and yp == 'X':
            tp += 1
    return tp

def true_negative(y_true, y_pred):
    """
    Function to calculate True Negatives
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: number of true negatives
    """
    tn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt != 'X' and yp != 'X':
            tn += 1
    return tn

def false_positive(y_true, y_pred):
    """
    Function to calculate False Positives
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: number of false positives
    """
    fp = 0
    for yt, yp in zip(y_true, y_pred):
        if yt != 'X' and yp == 'X':
            fp += 1
    return fp

def false_negative(y_true, y_pred):
    """
    Function to calculate False Negatives
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: number of false negatives
    """
    fn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 'X' and yp != 'X':
            fn += 1
    return fn

def securityMetrics(text, chunksExp, chunksPred):
    """
    Function that calculates Accuracy, Precision & Recall    
    """
    
    # expected
    list_of_words = chunksExp
    pat = "|".join(sorted(map(re.escape, list_of_words), key=len, reverse=True))
    pattern = re.compile(f'{pat}|(.)', re.S)
    y_true = list(pattern.sub(lambda m: " " if m.group(1) else len(m.group(0))*"X", text))

    # predicted
    list_of_words = chunksPred
    pat = "|".join(sorted(map(re.escape, list_of_words), key=len, reverse=True))
    pattern = re.compile(f'{pat}|(.)', re.S)
    y_pred = list(pattern.sub(lambda m: " " if m.group(1) else len(m.group(0))*"X", text))
    
    Accuracy = accuracy(y_true, y_pred)
    TP = true_positive(y_true, y_pred)
    TN = true_negative(y_true, y_pred)
    FP = false_positive(y_true, y_pred)
    FN = false_negative(y_true, y_pred)
    Precision = TP/(TP + FP)
    Recall = TP/(TP + FN)

    return Accuracy, Precision, Recall

schema =  StructType([
    StructField("accuracy", DoubleType(), False),
    StructField("precision", DoubleType(), False),
    StructField("recall", DoubleType(), False)
])
