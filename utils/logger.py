"""
Модуль для обеспечения логирования всего проекта.
"""

import logging


def get_stream_handler(log_format: str):
    """
    Функция для задания параметров логирования через создание хэндлера для
    сообщений логера.

    :param log_format: Строка с форматом для сообщений логгера.
    :return: Объект logging.StreamHandler.
    """
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(
        log_format,
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    return stream_handler


def get_logger(
        name: str,
        log_format: str = ("%(asctime)s.%(msecs)03d %(levelname)s in %(name)s: "
                           "%(message)s")

) -> logging.Logger:
    """
    Функция для создания объекта логера. Принимает на вход название контейнера,
    для которого нужно создать логер, и формат вывода сообщений для логера.

    :param name: Название контейнера, для которого нужно создать объект логера.
    :param log_format: Строка с форматом для сообщений логера.
    :return: Объект logging.Logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(get_stream_handler(log_format))
    return logger
