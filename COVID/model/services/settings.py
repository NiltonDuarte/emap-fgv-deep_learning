import logging
import sys

APP_NAME = 'EMAp/DL/COVID'
DATA_FOLDER = '/home/niltonduarte/workspace/FGV/DL/COVID/dataset/preprocessed'
MAX_AIO_TASKS = 15

MLFLOW_ENABLED = True
MLFLOW_TRACKING_SERVER = None

if MLFLOW_ENABLED:
    from google.cloud import secretmanager_v1 as secretmanager
    import json
    import os
    secretclient = secretmanager.SecretManagerServiceClient()
    mlflow_tracking_server_credentials_uri = 'projects/159165907154/secrets/mlflow_tracking_server_credential/versions/1'

    mlflow_tracking_server_credentials = json.loads(secretclient.access_secret_version(
        name=mlflow_tracking_server_credentials_uri).payload.data.decode('UTF-8'))
    MLFLOW_TRACKING_SERVER = mlflow_tracking_server_credentials["MLFLOW_TRACKING_SERVER"]

    # Setting environment variables for mlflow client
    os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_tracking_server_credentials["MLFLOW_TRACKING_USERNAME"]
    os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_tracking_server_credentials["MLFLOW_TRACKING_PASSWORD"]

log_format = logging.Formatter(
    '%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger(APP_NAME)
logger.setLevel(logging.DEBUG)

consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(log_format)
logger.addHandler(consoleHandler)

LOGGER = logger
