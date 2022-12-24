from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql import Window

import re, unicodedata

from langdetect import detect
from langid.langid import LanguageIdentifier, model
from googletrans import Translator
from sparknlp.base import *
from sparknlp.annotator import *

from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer as mlRegexTokenizer, NGram, HashingTF, MinHashLSH


def customTextNormalize(text, options):
    """
    Preprocesado de texto necesario antes de aplicar detectores o traductores
    """

    # texto en minúscula para el traductor
    if options['lowerstring']:
        text = str(text).lower()

    # soluciona casos donde unen cadenas con . o /
    if options['dotdashsplits']:
        text = ' '.join(text.split('.'))
        text = ' '.join(text.split('/'))

    # elimina palabras que contienen números
    if options['alfanumremoval']:
        regex = '\S*\d\S*'
        text = re.sub(regex, ' ', str(text))

    # elimina caracteres especiales excepto los propios del idioma
    if options['specialremoval']:
        regex = '[^A-Za-z\u00f1\u00d1áàâçéèêëíîïóôúûùüÿñæœâêôãõ ]+'
        text = re.sub(regex, '', str(text))

    # elimina palabras de una sola letra
    if options['singleletterremoval']:
        text = re.sub('(^|\s+)(\S(\s+|$))+', ' ', str(text))

    # elimina multiples espacios
    if options['stripstring']:
        text = re.sub('\s+', ' ', str(text))

    if options['nfd']:
        text = unicodedata.normalize('NFD', text)

    return str(text)


normalize_text_udf = lambda d: F.udf(lambda text: customTextNormalize(text, d), T.StringType())


def rename_columns(df, dict_rename):
    """
    Selecciona columnas requeridas de un DataFramne (Spark) y las renombra
    """

    columns = list(dict_rename.keys())
    df = df.select(columns)
    for col in columns:
        df = df.withColumnRenamed(col, dict_rename[col])

    return df


def filter_obsoletos(df):
    """
    Filtra aquellas spare parts que no se han marcado como obsoletas
    Los materiales que cumplan al menos una de las siguientes condiciones no serían relevantes:
    * Cross-Plant Material status – MARA-MSTAE:  diferente de S1
    * DF client level – MARA – LVORM:  marcado con flag
    IMPORTANTE: Esta función debe aplicarse despues del renombrado de campos
    """

    condition = (
            ((F.col('MSTAE') == 'S1') & (F.col('LVORM').isNull())) |
            ((F.col('MSTAE') == 'S1') & (F.col('LVORM') != 'X'))
    )

    return df.filter(condition)


def exact_match_brand_removal(data,
                              column_name_in,
                              column_name_out,
                              list_of_brands):
    """
    lee la lista de fabricantes a eliminar y los quita del texto
    spark:sparkContext
    data:dataframe
    column_name:columna a limpiar
    file:path con la lista de fabricantes
    """

    return data.withColumn(column_name_out, F.regexp_replace(column_name_in, '|'.join(list_of_brands), ''))


def minhash_signatures(df, descColumn, ngramLength=4, numHashTables=5, threshold=0.5):
    """
    Creates an output dataframe with similar items based on descColumn
    """

    columns = df.columns

    pipeline1 = Pipeline(stages=[
        mlRegexTokenizer(inputCol=descColumn, outputCol="tokens", minTokenLength=1, pattern=""),
        NGram(inputCol="tokens", outputCol="ngrams", n=ngramLength),
        HashingTF(inputCol="ngrams", outputCol="vectors")
    ])
    transformed_df = pipeline1.fit(df).transform(df)

    isNoneZeroVector = F.udf(lambda x: x.numNonzeros() > 0, T.BooleanType())
    transformed_df = transformed_df.filter(isNoneZeroVector(F.col("vectors")))

    pipeline = Pipeline(stages=[
        MinHashLSH(inputCol="vectors", outputCol="lsh", numHashTables=numHashTables, seed=1)
    ])

    model = pipeline.fit(transformed_df)
    transformed_df = model.transform(transformed_df)

    # similitud
    results = model.stages[-1].approxSimilarityJoin(transformed_df, transformed_df, threshold, distCol="dist_jaccard")

    results = results.select(*[F.col("datasetA." + c).alias(c) for c in columns],
                             *[F.col("datasetB." + c).alias("similar_" + c) for c in columns],
                             "dist_jaccard") \
        .filter(F.col("MATNR") != F.col("similar_MATNR"))

    return results


def minhash_signatures_multilang(df,
                                 descColumn,
                                 transColumn,
                                 ngramLength=4,
                                 numHashTables=5,
                                 threshold=0.5):
    """
    Creates an output dataframe with similar items based on descColumn and transColumn
    descColumn: columna con la descripción para traducir
    transColumn: columna con la descripción traducida
    """

    isNoneZeroVector = F.udf(lambda x: x.numNonzeros() > 0, T.BooleanType())

    df_org = df.select(list(set(df.columns) - set(transColumn)))
    df_tra = df.select(list(set(df.columns) - set(descColumn)))

    # fiteamos la pipeline al original y la usamos para transformar ambos dfs
    # esto nos ayudara a encontrar patrones de df_org en df_tra
    pipeline1 = Pipeline(stages=[
        mlRegexTokenizer(inputCol=descColumn, outputCol="tokens", minTokenLength=1, pattern=""),
        NGram(inputCol="tokens", outputCol="ngrams", n=ngramLength),
        HashingTF(inputCol="ngrams", outputCol="vectors")
    ])

    # fit model to original
    model1 = pipeline1.fit(df_org)
    df_org_2 = model1.transform(df_org) \
                     .filter(isNoneZeroVector(F.col("vectors")))

    # transform translated
    df_tra_2 = model1.transform(df_tra) \
                     .filter(isNoneZeroVector(F.col("vectors")))

    # con los elementos nulos eliminados aplico minhashing
    pipeline = Pipeline(stages=[
        MinHashLSH(inputCol="vectors", outputCol="lsh", numHashTables=numHashTables, seed=1)
    ])

    model = pipeline.fit(df_org_2)
    df_org_2 = model.transform(df_org_2)
    df_tra_2 = model.transform(df_tra_2)

    # similitud entre el original y el traducido
    results = model.stages[-1].approxSimilarityJoin(df_org_2, df_tra_2, threshold, distCol="dist_jaccard")

    results = results.select(*[F.col("datasetA." + c).alias(c) for c in df_org_2.columns],
                             *[F.col("datasetB." + c).alias("similar_" + c) for c in df_tra_2.columns],
                             "dist_jaccard") \
        .filter(F.col("MATNR") != F.col("similar_MATNR"))

    return results


def prepare_output(results, data):
    """
    prepara los resultados que genera la función minhash_signatures para la salida a producción
    :return:
    """

    # Por cada MATNR encuentra el elemento con la mínima distancia de jaccard y elimina el resto
    # (este paso quizás haya que refinarlo y poner que colecte un array de elementos!)
    w = Window.partitionBy("MATNR")
    results = results.withColumn('min_dist_jaccard', F.min('dist_jaccard').over(w))\
                     .filter(F.col('dist_jaccard') == F.col('min_dist_jaccard'))\
                     .drop("min_dist_jaccard")\
                     .dropDuplicates(["MATNR"])\
                     .orderBy(F.col("dist_jaccard").asc())

    possible_missing = data.dropDuplicates(["MATNR"]) \
                            .withColumn("similar_row_id", F.lit(None).cast(T.LongType())) \
                            .withColumn("similar_MAKTX", F.lit(None).cast(T.StringType())) \
                            .withColumn("similar_MATNR", F.lit(None).cast(T.StringType())) \
                            .withColumn("similar_SPRAS_ISO", F.lit(None).cast(T.StringType())) \
                            .withColumn("similar_ERNAM", F.lit(None).cast(T.StringType())) \
                            .withColumn("similar_WERKS", F.lit(None).cast(T.ArrayType(T.StringType()))) \
                            .withColumn("similar_MATKL", F.lit(None).cast(T.StringType())) \
                            .withColumn("dist_jaccard", F.lit(None).cast(T.DoubleType()))

    possible_missing = possible_missing.join(results, "MATNR", "leftanti")

    return results.unionByName(possible_missing)


#
# Detectores de lenguaje
#

def google_language_detector(text):
    """
    :text: normalized text
    :return: string language detected
    """

    translator = Translator()

    if len(text) <= 2:
        lg = 'Unknown'
    else:
        lg = translator.detect(text).lang

    return lg


google_language_detector_udf = F.udf(google_language_detector, T.StringType())


def spacy_language_detector(text, detector='langid'):
    """
    :text: normalized text
    :detector: spacy model
    :return: string language detected
    """

    lang_identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)

#    nlp = spacy.load("en_core_web_sm")
#    nlp.add_pipe("language_detector")

    if len(text) <= 2:
        lg = 'Unknown'
    elif detector == 'langid':
        lg = lang_identifier.classify(text)[0]
    elif detector == 'langdetect':
        lg = detect(text)
#    elif detector == 'spacy_langdetect':
#        lg = nlp(text)._.language

    return lg


spacy_language_detector_udf = F.udf(spacy_language_detector, T.StringType())


def sparkNLP_language_detector(df, inputColumn, outputColumn, modelpath, threshold):
    """
    sparkNLP pipeline for language detection
    :param df:
    :param inputColumn: name of the column to apply the detector
    :param outputColumn: output column
    :return: df with language detected in new columns
    """

    documentAssembler = DocumentAssembler()\
        .setInputCol(inputColumn)\
        .setOutputCol("document")

    language_detector = LanguageDetectorDL.load(modelpath) \
        .setInputCols(["document"])\
        .setOutputCol('result')\
        .setThreshold(threshold)\
        .setCoalesceSentences(True)

    languagePipeline = LightPipeline(Pipeline(stages=[
        documentAssembler,
        language_detector
    ]))

    return languagePipeline.fit(df).transform(df) \
                .withColumn(outputColumn, F.expr('result.result[0]')) \
                .drop('document', 'result')


def sparkNLP_english_translator(df, inputColumn, outputColumn, modelpath):
    """
    sparkNLP pipeline for language translation
    :param df:
    :param inputColumn: name of the column to apply the detector
    :param outputColumn: output column
    :return: df with description translated to english
    """

    documentAssembler = DocumentAssembler() \
        .setInputCol(inputColumn) \
        .setOutputCol("document")

    marian = MarianTransformer.load(modelpath) \
        .setInputCols(["document"]) \
        .setOutputCol('result')

    translationPipeline = Pipeline(
        stages=[
            documentAssembler,
            marian
        ])

    return translationPipeline.fit(df).transform(df) \
                .withColumn(outputColumn, F.expr('result.result[0]')) \
                .drop('document', 'result')









