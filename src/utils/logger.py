import logging


def setup_logger(name: str, level: int = logging.INFO):
    logger = logging.getLogger(name)
    log_file = f"reports/logs/{name}.log"

    if not logger.handlers:
        logger.setLevel(level)

        # Create stdout handler for logging to the console (logs all five levels)
        stdout_handler = logging.StreamHandler()
        stdout_handler.setLevel(logging.DEBUG)

        # Create file handler for logging to a file (logs all five levels)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Define format for logs
        formatter = logging.Formatter(
            fmt="{asctime} - {levelname} - {funcName}:{lineno} - {message}",
            style="{",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        stdout_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        logger.addHandler(stdout_handler)
        logger.addHandler(file_handler)

    return logger
