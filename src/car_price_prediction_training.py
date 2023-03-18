from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StructType, StringType, IntegerType
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler, IndexToString
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from util.applogger import getAppLogger
from util.appconfigreader import AppConfigReader
import os

def loadCsvFiles(spark, schema, csvDir):
    '''Loads CSV files from the given path and returns a dataframe
    Args: spark - Active spark session
          schema - Schema of the data in csv files
          csvDir - Path of csv files
    '''
    try:
        loadedData = spark.read.format('csv').schema(schema).load(csvDir)
        return loadedData
    except Exception as error:
        logger.exception(f"Failed to load csv files {error}")
        raise
    else:
        logger.info("CSV files are loaded successfully")

def createPipeline(carData, lrElasticNetParam, lrRegParam):
    '''Creates a pipeline for converting the data into features and label with the required format
    Args: carData - Input data for the feature and label processing
          lrElasticNetParam - ElasticNet parameter of LR, 0-L2 penalty and 1-L1 penalty
          lrRegParam - Regularization parameter
    '''

    # convert the feature string column to numeric value using simple string indexer
    featureIndexer = [StringIndexer().setInputCol(col).setOutputCol(col+"_indexer").fit(carData)
                      for col in ['maintenance','lug_boot_size', 'safety','class']]

    # convert the labels into numeric value
    labelIndexer = StringIndexer().setInputCol('buying').setOutputCol('label').fit(carData)

    # merge all the feature columns into a vector column
    va = VectorAssembler(inputCols=['maintenance_indexer','door',
                                    'lug_boot_size_indexer','safety_indexer', 'class_indexer'],
                         outputCol='vec_features')

    # standardizes feature on the merged features data
    ss = StandardScaler().setInputCol(va.getOutputCol()).setOutputCol('features').fit(va.transform(carData))

    # lr model set feature col which is standardized
    lr = LogisticRegression().setFeaturesCol('features')

    # convert the numeric label to string value
    labelConverter = IndexToString(inputCol='prediction', outputCol='predictedLabel', labels=labelIndexer.labels)

    # build stages of preprocessing and model build
    stages = [featureIndexer, labelIndexer, va, ss, lr, labelConverter]

    # build the pipeline with the stages
    pipeline = Pipeline().setStages(stages)

    # build several hyper-parameters combination used for the model training
    params = ParamGridBuilder().addGrid(lr.elasticNetParam,lrElasticNetParam)\
        .addGrid(lr.regParam,lrRegParam).build()

    # evaluate the different models using lrMetric
    evaluator = MulticlassClassificationEvaluator(labelCol='label',
                                                  predictionCol='prediction', metricName=lrMetric)

    return pipeline, params, evaluator

def trainModels(trainData, pipeline, params, evaluator, numFolds=3):
    '''Train the models and results it
    Args: trainData - Data to train LR model
          pipeline - Pipeline with set of transformers and estimators
          params - List of parameters used for the model tuning
          evaluator - Evaluates the model
          numFolds - Number of splitting data into a set of folds
    '''

    # perform model training by split the dataset and capture the metrics of each
    crossValidator = CrossValidator(estimator=pipeline, estimatorParamMaps=params,
                                    evaluator=evaluator, numFolds=numFolds)
    cvModels = crossValidator.fit(trainData)
    return cvModels

if __name__ == '__main__':
    try:
        logger = getAppLogger(__name__)

        # Read application CFG file
        logger.info('Read application parameters from config file')
        appConfigReader = AppConfigReader()
        if 'APP' in appConfigReader.config:
            appCfg = appConfigReader.config['APP']
            inputDir = appCfg['CSV_DIR']
            lrElasticNetParam = [float(n) for n in appCfg['LR_ELASTIC_NET_PARAM'].split(',')]
            lrRegParam = [float(n) for n in appCfg['LR_REG_PARAM'].split(',')]
            lrMetric = appCfg['LR_METRIC_NAME']
            numFolds = int(appCfg['DATA_SPLIT_COUNT'])
            modelDir = appCfg['LR_MODEL_DIR']
        else:
            logger.error('Application details are missed out to configure')
            raise SystemExit(1)

        # create a spark session
        logger.info('Create a spark session')
        spark = SparkSession.builder.appName('CarPricePrediction').getOrCreate()
        if spark:
            # set the schema for the dataset
            manualSchema = StructType([StructField('buying', StringType(), False), StructField('maintenance', StringType(), False),
                                       StructField('doors', IntegerType(), False), StructField('person', IntegerType(), False),
                                       StructField('lug_boot_size', StringType(), False), StructField('safety', StringType(), False),
                                       StructField('class', StringType(), False)])
            # load the data set into dataframe
            logger.info('Load the csv files form the input directory')
            carData= loadCsvFiles(spark, manualSchema, inputDir)
            carColumns = carData.columns

            # split the dataset into train and test
            carTrainData, carTestData = carData.randomSplit([0.7, 0.3])

            # create the pipeline for preprocessing, build  hyper-parameters combinations
            # and evaluator for the model selection
            logger.info('Create a pipeline with the set of transformer and estimators')
            pipeline, params, evaluator = createPipeline(carData, lrElasticNetParam, lrRegParam)

            # train the model with different combination of the parameters
            logger.info('Train and tune the mode with the parameters configured')
            cvModels = trainModels(carTrainData, pipeline, params, evaluator, numFolds)

            # select the best model from the trained models and save it in disk for prediction
            lrModel = cvModels.bestModel.stages[3]
            logger.info('Save the best model for the further usage')
            cvModels.bestModel.write().overwrite().save(modelDir)

            # logs some metrics about the best model
            logger.info('Details about the trained model')
            logger.info(f'Evaluation based on the metric {lrMetric} is {evaluator.evaluate(cvModels.transform(carTestData))}')
            logger.info(f'Precision: {lrModel.summary.weightedPrecision}')
            logger.info(f'Recall: {lrModel.summary.weightedRecall}')
            logger.info(f'Weighted F Score: {lrModel.summary.weightedFMeasure()}')
            logger.info(f'True Positive Rate: {lrModel.summary.weightedTruePositiveRate}')
            logger.info(f'False Positive Rate: {lrModel.summary.weightedFalsePositiveRate}')
            logger.info(f'Parameters used: {lrModel.extractParamMap()}')
    except Exception as error:
        logger.exception(f'Something went wrong here {error}')
    else:
        logger.info('Model has been trained and ready for any new prediction')
    finally:
        spark.stop()
