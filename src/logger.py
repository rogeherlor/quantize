# src/logger.py
import logging

logger = logging.getLogger("src")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # file_handler = logging.FileHandler("src.log")
    # file_handler.setFormatter(formatter)
    # logger.addHandler(file_handler)