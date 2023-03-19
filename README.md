# car_price_prediction

SPARK ML library is used to train and build the model.

Trains a logistic regression model using the dataset `https://archive.ics.uci.edu/ml/datasets/Car+Evaluation`.

Use spark-submit to run the script, provided adding the pyspark installation directory to PATH variable and have the soure code path in the variable PYTHON_PATH

### Model Training
Command: `spark-submit --master [host/yarn] --py-files util.zip --files car_price_prediction.properties car_price_prediction_training.py`

Note: util.zip file contains logging and config reader modules.

Reads the data in the CSV file from the input path, tune it with the parameters configured then the best model is saved for further predictions. Features used are maintenance, no_of_doors, lug_boot_size, safety and car_class. This contians 4 labels vhigh, high, med and low to predict.

Pipeline is used to build the transformation of data which will be fed to the model creation. The model is tuned with the set of parameters and chosen the best one using an evaluator with the metric configured. Saves the best model in the disk for the future predictions.

Predict Car Price
Command: `spark-submit --master [host/yarn] --py-files util.zip --files car_price_prediction.properties car_price_prediction.py [maintenance] [no_of_doors] [lug_boot_size] [safety] [car_class]`

Note: util.zip file contains logging and config reader modules.

This predicts the car buying category from the features passed as argument. The predicted category is mapped to the price range configured in the yaml file `price_mapping.yaml`

Example: `car_price_prediction.py high 4 big high good`

Prerequisite: The model should be trained and available in the disk for loading so the model training script needs to execute first.
