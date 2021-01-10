import asyncio

from services.mlflow import MLFlow
from services.torch_model import COVIDModel
from services.settings import (
    MLFLOW_ENABLED, MLFLOW_TRACKING_SERVER,
    DATA_FOLDER)

"""
3.1 Experiment by changing the activation  functions (Tanh, ReLU) and analyze the results.
3.2 Experiment changing regularization strategies (1. without Regularization. 2 with Batch Normalization. 3 with Dropout)
3.3 With  a CNN with 3 layer, use kernel size:
 (7x7,5x5,3x3)
 (5x5,5x5,3x3)
 (3x3,3x3,3x3)
3.4 Conclude from the results
"""


def grid_search():
    for kernel_sizes in [(7, 5, 3), (5, 5, 3), (3, 3, 3)]:
        for activation_function in ['Tanh', 'ReLU']:
            yield kernel_sizes, activation_function


for kernel_sizes, activation_function in grid_search():
    model = COVIDModel(
        activation_function=activation_function,
        kernel_sizes=kernel_sizes)
    model.load_images(DATA_FOLDER)
    list_loss = model.train()

    if MLFLOW_ENABLED:
        model_id = f'{activation_function}/{kernel_sizes}'
        mlflow = MLFlow()
        mlflow.setup_mlflow(mlflow_server=MLFLOW_TRACKING_SERVER,
                            mlflow_experiment_name='emap_dl', model_id=model_id)

        async def log_and_finish(list_loss):
            await mlflow.async_log('params', {'activation_function': activation_function,
                                              'kernel_sizes': kernel_sizes})
            await mlflow.async_log('metrics', list_loss, prefix='training_')
            await mlflow.end_run()

        asyncio.run(log_and_finish(list_loss))
