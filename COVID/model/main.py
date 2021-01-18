import asyncio

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
    for trial in range(3):
        for activation_function in ['Tanh', 'ReLU']:
            for regularization_strategy in ['without', 'BatchNorm2d', 'Dropout']:
                for kernel_sizes in [(7, 5, 3), (5, 5, 3), (3, 3, 3)]:
                    yield activation_function, regularization_strategy, kernel_sizes, trial


index = -1
# change this to skip some trials
skip_trials = 0
for activation_function, regularization_strategy, kernel_sizes, trial in grid_search():
    index += 1
    if index < skip_trials:
        print(
            f'Skipping {activation_function} / {kernel_sizes} / {regularization_strategy}')
        continue
    model = COVIDModel(
        activation_function=activation_function,
        regularization_strategy=regularization_strategy,
        kernel_sizes=kernel_sizes)
    model.load_images(DATA_FOLDER)
    list_loss = model.train(num_epochs=4)
    eval_metrics = model.get_eval_metrics()
    test_metrics = model.get_test_metrics()
    # Test the model

    if MLFLOW_ENABLED:
        from services.mlflow import MLFlow
        model_id = f'{activation_function}/{kernel_sizes}/{regularization_strategy}/{trial}'
        mlflow = MLFlow()
        mlflow.setup_mlflow(mlflow_server=MLFLOW_TRACKING_SERVER,
                            mlflow_experiment_name='emap_dl', model_id=model_id)

        async def log_and_finish(list_loss):
            await mlflow.async_log('params', {'activation_function': activation_function,
                                              'kernel_sizes': kernel_sizes,
                                              'regularization_strategy': regularization_strategy})
            await mlflow.async_log('metrics', list_loss, prefix='training.')
            await mlflow.async_log('metrics', eval_metrics,
                                   prefix='eval.')
            await mlflow.async_log('metrics', test_metrics,
                                   prefix='test.')
            await mlflow.end_run()

        asyncio.run(log_and_finish(list_loss))
    else:
        print('****************')
        print('* EVAL METRICS *')
        print('****************')
        print(eval_metrics)
        print('****************')
        print('* TEST METRICS *')
        print('****************')
        print(test_metrics)
