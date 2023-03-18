from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline


def convertFeatures(data, cols):
    '''Loads CSV files from the given path and returns a dataframe
    Args: data - dataframe
          cols - list of feature columns to be converted to numeric
          csvDir - Path of csv files
    '''
    # convert the feature string column to numeric value using simple string indexer
    featureIndexer = [StringIndexer().setInputCol(col).setOutputCol(col + "_indexer")
                      for col in cols]

    featurePipeline = Pipeline(stages=featureIndexer)

    return featurePipeline.fit(data).transform(data)
