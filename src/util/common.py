from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler, IndexToString, StandardScaler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from util.applogger import getAppLogger



logger = getAppLogger(__name__)


def convertFeatures(data, cols):
    """
    converts the string feature columns to numeric
    :param data: data
    :param cols: list of columns to be converted
    :return: dataframe with the numeric feature columns added "_indexer"
    """
    # convert the feature string column to numeric value using simple string indexer
    featureIndexer = [StringIndexer().setInputCol(col).setOutputCol(col + "_indexer")
                      for col in cols]

    featurePipeline = Pipeline(stages=featureIndexer)

    return featurePipeline.fit(data).transform(data)


def predictDataToDF(spark, args, column_names):
    """
    converts the raw data into spark dataframe which will be used for prediction from the saved model
    :param spark: spark session
    :param args: list of data to be added to dataframe
    :param column_names: column names
    :return: spark dataframe
    """
    return spark.createDataFrame([(args.maintenance, args.doors, args.lug_boot_size, args.safety, args.car_class)],
                                 column_names)


def carPricePrediction(dataToPredict, modelDir):
    """
    predicts the car price for the data passed
    :param dataToPredict: data to be predicted
    :param modelDir: directory where the trained model is saved
    :return:
    """

    # load the data pre-processing pipeline from the disk
    pipelineModel = PipelineModel.load(modelDir)
    if pipelineModel:
        # merge all features
        vaData = pipelineModel.stages[1].transform(dataToPredict)
        # standardize the feature data
        modelData = pipelineModel.stages[2].transform(vaData)
        # set the feature column for the model
        lrModel = pipelineModel.stages[3]
        # convert the numeric label to actual value
        indexString = pipelineModel.stages[4]

        if vaData and modelData and lrModel and indexString:
            logger.info('Model has been loaded and going to predict for the values passed')
            prediction = lrModel.transform(modelData).select('prediction')
            predictedValue = indexString.transform(prediction).collect()
            return predictedValue[0].predictedLabel
        else:
            logger.error(f'Problem in loading the one of the stages from the model saved in the location {modelDir}')
            raise SystemExit(3)
    else:
        logger.error(f'Problem in loading the model from the directory {modelDir}')
        raise SystemExit(3)


def trainModels(trainData, pipeline, params, evaluator, numFolds=3):
    """Train the models and results it
    Args: trainData - Data to train LR model
          pipeline - Pipeline with set of transformers and estimators
          params - List of parameters used for the model tuning
          evaluator - Evaluates the model
          numFolds - Number of splitting data into a set of folds
    """

    # perform model training by split the dataset and capture the metrics of each
    crossValidator = CrossValidator(estimator=pipeline, estimatorParamMaps=params,
                                    evaluator=evaluator, numFolds=numFolds)
    cvModels = crossValidator.fit(trainData)
    return cvModels


def createCarPredictionPipeline(carData, featureStrColumns, lrElasticNetParam, lrRegParam, lrMetric):
    """
    Creates a pipeline for converting the data into features and label with the required format
    :param carData: Input data for the feature and label processing
    :param featureStrColumns: Feature column of type string
    :param lrElasticNetParam: ElasticNet parameter of LR, 0-L2 penalty and 1-L1 penalty
    :param lrRegParam: Regularization parameter
    :param lrMetric: Metric name
    :return: Model pipeline
    """

    # convert the labels into numeric value
    labelIndexer = StringIndexer().setInputCol('buying').setOutputCol('label').fit(carData)

    # merge all the feature columns into a vector column
    va = VectorAssembler(inputCols=[col + '_indexer' for col in featureStrColumns],
                         outputCol='vec_features')

    # standardizes feature on the merged features data
    ss = StandardScaler().setInputCol(va.getOutputCol()).setOutputCol('features').fit(va.transform(carData))

    # lr model set feature col which is standardized
    lr = LogisticRegression().setFeaturesCol('features')

    # convert the numeric label to string value
    labelConverter = IndexToString(inputCol='prediction', outputCol='predictedLabel', labels=labelIndexer.labels)

    # build stages of preprocessing and model build
    stages = [labelIndexer, va, ss, lr, labelConverter]

    # build the pipeline with the stages
    pipeline = Pipeline().setStages(stages)

    # build several hyper-parameters combination used for the model training
    params = ParamGridBuilder().addGrid(lr.elasticNetParam, lrElasticNetParam) \
        .addGrid(lr.regParam, lrRegParam).build()

    # evaluate the different models using lrMetric
    evaluator = MulticlassClassificationEvaluator(labelCol='label',
                                                  predictionCol='prediction', metricName=lrMetric)

    return pipeline, params, evaluator


def loadCsvFiles(spark, schema, csvDir):
    """
    Loads CSV files from the given path and returns a dataframe
    :param spark: Spark session
    :param schema: Schema of the data
    :param csvDir: Path of the source file
    :return: spark dataframe
    """
    try:
        loadedData = spark.read.format('csv').schema(schema).load(csvDir)
        return loadedData
    except Exception as err:
        logger.exception('Failed to load csv files')
        raise
    else:
        logger.info("CSV files are loaded successfully")