from configparser import ConfigParser
import os


class AppConfigReader():
    '''Loads the config file values'''

    def __init__(self):
        self.config = ConfigParser()
        # Get the config file path from environmental variable PY_APP_CONFIG
        # cfgDir = os.environ.get('CFG_DIR')
        # if cfgDir:
        #     cfgFile = cfgDir + "\\car_price_prediction.properties"
        # else:
        cfgFile = "E:\\Python\\github\\car_price_prediction\\config\\car_price_prediction.properties"


        # Load the CFG file
        self.config.read(cfgFile)
