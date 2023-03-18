from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StructType, StringType
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler, IndexToString
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from util.applogger import getAppLogger
from util.appconfigreader import AppConfigReader
from util.common import convertFeatures, createCarPredictionPipeline, loadCsvFiles, trainModels


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
            featureStrColumns = [str(col) for col in appCfg.get('FEATURE_STRING_COLS', None).split(',')]
        else:
            logger.error('Application details are missed out to configure')
            raise SystemExit(1)

        # create a spark session
        logger.info('Create a spark session')
        spark = SparkSession.builder.appName('CarPricePrediction').getOrCreate()
        if spark:
            # set the schema for the dataset
            manualSchema = StructType(
                [StructField('buying', StringType(), False), StructField('maintenance', StringType(), False),
                 StructField('doors', StringType(), False), StructField('person', StringType(), False),
                 StructField('lug_boot_size', StringType(), False), StructField('safety', StringType(), False),
                 StructField('car_class', StringType(), False)])
            # load the data set into dataframe
            logger.info('Load the csv files form the input directory')
            carData = loadCsvFiles(spark, manualSchema, inputDir)
            carColumns = carData.columns

            # feature columns converted into numeric values
            if featureStrColumns:
                carData = convertFeatures(carData, featureStrColumns)

            # split the dataset into train and test
            carTrainData, carTestData = carData.randomSplit([0.7, 0.3])

            # create the pipeline for preprocessing, build  hyper-parameters combinations
            # and evaluator for the model selection
            logger.info('Create a pipeline with the set of transformer and estimators')
            pipeline, params, evaluator = createCarPredictionPipeline(carData, featureStrColumns,
                                                                      lrElasticNetParam, lrRegParam, lrMetric)

            # train the model with different combination of the parameters
            logger.info('Train and tune the mode with the parameters configured')
            cvModels = trainModels(carTrainData, pipeline, params, evaluator, numFolds)

            # select the best model from the trained models and save it in disk for prediction
            lrModel = cvModels.bestModel.stages[3]
            logger.info('Save the best model for the further usage')
            cvModels.bestModel.write().overwrite().save(modelDir)

            # logs some metrics about the best model
            logger.info('Details about the trained model')
            logger.info(
                f'Evaluation based on the metric {lrMetric} is {evaluator.evaluate(cvModels.transform(carTestData))}')
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
