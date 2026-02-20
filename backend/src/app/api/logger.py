import logging

def get_route_logger():
    logger = logging.getLogger("routes")
    logger.setLevel(logging.INFO)

    if not logger.handlers:  #
        file_handler = logging.FileHandler("routes.log")
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
