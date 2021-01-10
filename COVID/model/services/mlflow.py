import mlflow
import asyncio as aio
import functools
from .settings import MLFLOW_ENABLED, LOGGER, MAX_AIO_TASKS


def prepend_key(data_obj: object, prefix: str):
    if isinstance(data_obj, dict):
        prepended_dict = {}
        for key in data_obj.keys():
            prepended_dict[prefix + str(key)] = data_obj.get(key)
        return prepended_dict

    if isinstance(data_obj, str):
        return prefix + data_obj

    if isinstance(data_obj, list):
        return [prepend_key(list_obj, prefix) for list_obj in data_obj]


def truncate_log_param(data_obj: object):
    if isinstance(data_obj, dict):
        copy_dict = {}
        for key in data_obj.keys():
            copy_dict[key] = str(data_obj[key])[:250]
        return copy_dict

    if isinstance(data_obj, str):
        return data_obj[:250]


def force_async(fn):
    """
    Turns a sync function to async function using asyncio
    """
    import asyncio

    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()

        partial_func = functools.partial(fn, *args, **kwargs)
        return await loop.run_in_executor(None, partial_func)

    return wrapper


class MLFlow:
    MAX_AIO_TASKS = MAX_AIO_TASKS

    def __init__(self, *, mlflow_logging_enabled=True, mlflow_param_logging_enabled=True, mlflow_metric_logging_enabled=True):
        self.mlflow_logging_enabled = mlflow_logging_enabled
        self.mlflow_param_logging_enabled = mlflow_param_logging_enabled
        self.mlflow_metric_logging_enabled = mlflow_metric_logging_enabled
        self.mlflow_artifact_logging_enabled = False
        self.logger = LOGGER
        self.aio_tasks = set()

    def setup_mlflow(self, mlflow_server, mlflow_experiment_name, model_id):
        mlflow.set_tracking_uri(mlflow_server)
        mlflow.set_experiment(mlflow_experiment_name)
        run = mlflow.start_run(run_name=model_id)
        run_id = run.info.run_id
        return run_id

    async def end_run(self):
        if len(self.aio_tasks) > 0:
            self.fprint("Waiting async tasks")
            _done, self.aio_tasks = await aio.wait(self.aio_tasks, return_when=aio.ALL_COMPLETED)
            self.raise_task(_done)

        self._end_run()

    def _end_run(self):
        mlflow.end_run()
        self.fprint("Run finished.")

    def raise_task(self, task_set):
        for task in task_set:
            if task.exception():
                raise task.exception()

    def fprint(self, text, end='\n', log_type='info'):
        getattr(self.logger, log_type)(text)

    async def async_log(self, _type, log_obj, prefix=''):
        # self.fprint(f"Addeding log to async tasks.")

        if isinstance(log_obj, list):
            log_obj = log_obj.copy()
            for list_obj in log_obj:
                await self.async_log(_type, list_obj, prefix)
            return
        if isinstance(log_obj, dict):
            log_obj = log_obj.copy()

        if len(self.aio_tasks) >= self.MAX_AIO_TASKS:
            # self.fprint(f"Tasks set full, waiting any task to complete")
            _done, self.aio_tasks = await aio.wait(
                self.aio_tasks, return_when=aio.FIRST_COMPLETED)
            self.raise_task(_done)

        self.aio_tasks.add(aio.create_task(
            self._async_log(_type, log_obj, prefix)))

    @force_async
    def _async_log(self, _type, log_obj, prefix):
        self.__log(_type, log_obj, prefix)

    def __log(self, _type, log_obj, prefix=''):
        self.fprint(f"Logging {_type}...")

        if not self.mlflow_logging_enabled:
            self.fprint(log_obj)
            return

        if _type == 'params' and self.mlflow_param_logging_enabled:
            log_obj = prepend_key(log_obj, prefix)
            log_obj = truncate_log_param(log_obj)
            mlflow.log_params(log_obj)

        if _type == 'metrics' and self.mlflow_metric_logging_enabled:
            log_obj['metrics'] = prepend_key(log_obj['metrics'], prefix)
            mlflow.log_metrics(**log_obj)

        if _type == 'artifact' and self.mlflow_artifact_logging_enabled:
            raise NotImplementedError()
            # mlflow.log_artifact(log_obj)

        if _type == 'artifacts' and self.mlflow_artifact_logging_enabled:
            raise NotImplementedError()
            # mlflow.log_artifacts(log_obj)

    def set_tags(self, **tags):
        self.fprint('Setting tags...')
        if not self.mlflow_logging_enabled:
            self.fprint(tags)
            return
        mlflow.set_tags(tags)
