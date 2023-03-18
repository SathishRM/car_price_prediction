from pyspark.sql import SparkSession
import argparse
import yaml
from util.applogger import getAppLogger
from util.appconfigreader import AppConfigReader
from util.common import convertFeatures, predictDataToDF, carPricePrediction


if __name__ == '__main__':
    try:
        logger = getAppLogger(__name__)

        # Read application CFG file
        logger.info('Read application parameters from config file')
        appConfigReader = AppConfigReader()
        if 'APP' in appConfigReader.config:
            appCfg = appConfigReader.config['APP']
            modelDir = appCfg['LR_MODEL_DIR']
            featureStrColumns = [str(col) for col in appCfg.get('FEATURE_STRING_COLS', None).split(',')]

            # get price range for each the buying category
            mappingFile=appCfg.get('PRICE_MAPPING_FILE')
            if mappingFile:
                with open(mappingFile) as f:
                    priceRange = yaml.load(f,yaml.FullLoader).get('predict_values')
            logger.info(f"price range f{priceRange} and type-{type(appCfg)}")
        else:
            logger.error('Application details are missed out to configure')
            raise SystemExit(1)

        #Parse the arguments passed
        argParser = argparse.ArgumentParser()
        argParser.add_argument('maintenance', type=str, help="car maintenance category")
        argParser.add_argument('doors', type=str, help="no of doors in the car")
        argParser.add_argument('lug_boot_size', type=str, help="boot size of the car")
        argParser.add_argument('safety', type=str, help="category of the safety standards")
        argParser.add_argument('car_class', type=str, help="category of the car class")
        args = argParser.parse_args()
        logger.info(f'Arguments received are maintenance:{args.maintenance}, doors:{args.doors}, '
                    f'lug_boot_size:{args.lug_boot_size}, safety:{args.safety}, class:{args.car_class}')

        # validate the arguments passed
        if args.maintenance and args.doors and args.lug_boot_size and args.safety and args.car_class:
            # create a spark session
            logger.info('Create a spark session')
            spark = SparkSession.builder.appName('CarPricePrediction').getOrCreate()
            if spark:
                # frame df from raw data
                dataToPredict = predictDataToDF(spark, args, featureStrColumns)
                # convert the features if required
                if featureStrColumns:
                    dataToPredict = convertFeatures(dataToPredict, featureStrColumns)

                # predicts car price
                logger.info('Predicts the car price for the given values...')
                predictedValue = carPricePrediction(dataToPredict, modelDir)
                logger.info(f'Value predicted is {priceRange.get(predictedValue)}')
        else:
            logger.error(f'Missing some required parameters, please checkt the script help for the list of arguments required')
            raise SystemExit(2)
    except Exception as error:
        logger.exception(f'Something went wrong here {error}')
    else:
        logger.info('Prediction has been completed')
    finally:
        spark.stop()
