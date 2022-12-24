import pyspark.sql.functions as F
import pyspark.sql.types as T

from pyspark.ml import Pipeline, functions as mlF
from pyspark.ml.feature import RegexTokenizer, NGram, HashingTF, MinHashLSH


def pyspark_transform(spark, df, param_dict):
    """
    :param spark: SparkSession
    :param df: Input DataFrame
    :param param_dict: Input dictionary
    :return: Transformed DataFrame
    """

    def MinHashLSHSignatures(df,
                            inputCol="concatenated",
                            outputCol="lsh",
                            numHashTables=5):

        important_cols = ["id_usuario", "tv_id"]

        # hashing model
        model = Pipeline(stages=[
            RegexTokenizer(pattern="", inputCol=inputCol, outputCol="tokens", minTokenLength=1),
            NGram(n=4, inputCol="tokens", outputCol="ngrams"),
            HashingTF(inputCol="ngrams", outputCol="vectors", ),
            MinHashLSH(inputCol="vectors", outputCol=outputCol, numHashTables=numHashTables)
        ])

        df_hashed = model.fit(df).transform(df)

        to_array = F.udf(lambda vv: [v.toArray().tolist()[0] for v in vv], T.ArrayType(T.FloatType()))
        df_hashed = df_hashed.select(*important_cols, to_array("lsh").alias("lsh")) \
            .select(important_cols + [F.col("lsh")[i].alias('lsh_{}'.format(i)) for i in range(numHashTables)])

        return df_hashed

    return MinHashLSHSignatures(df)
