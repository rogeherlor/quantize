# src/logger.py
import logging

logger = logging.getLogger("src_logger")
logger.setLevel(logging.DEBUG)
logger.propagate = False

if not logger.handlers:
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler("src.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

# Performance logger with simplified formatting
logger_performance = logging.getLogger("performance_logger")
logger_performance.setLevel(logging.INFO)
logger_performance.propagate = False

if not logger_performance.handlers:
    perf_stream_handler = logging.StreamHandler()
    perf_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    perf_stream_handler.setFormatter(perf_formatter)
    logger_performance.addHandler(perf_stream_handler)

    perf_file_handler = logging.FileHandler("performance.log")
    perf_file_handler.setFormatter(perf_formatter)
    logger_performance.addHandler(perf_file_handler)