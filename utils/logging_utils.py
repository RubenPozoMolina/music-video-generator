import logging
import time
import platform

import psutil


def show_elapsed_time(start_time):
    elapsed_time = time.perf_counter() - start_time
    hours, remainder = divmod(int(elapsed_time), 3600)
    minutes, seconds = divmod(remainder, 60)
    logging.info(f"Elapsed time: {hours}h {minutes}m {seconds}s")


def log_system_info():
    logging.info("Starting System Information Check...")

    try:
        # --- CPU Information ---
        cpu_model = platform.processor()
        logical_cores = psutil.cpu_count(logical=True)
        physical_cores = psutil.cpu_count(logical=False)
        cpu_usage = psutil.cpu_percent(interval=1)

        logging.info(f"CPU Model: {cpu_model if cpu_model else 'N/A'}")
        logging.info(f"Cores: {physical_cores} Physical / {logical_cores} Logical")
        logging.info(f"Current Total CPU Usage: {cpu_usage}%")

        # --- RAM Information ---
        svmem = psutil.virtual_memory()
        # Conversion: Bytes to GB (2^30)
        total_ram = svmem.total / (1024 ** 3)
        available_ram = svmem.available / (1024 ** 3)

        logging.info(f"RAM Total: {total_ram:.2f} GB")
        logging.info(f"RAM Available: {available_ram:.2f} GB")
        logging.info(f"RAM Usage Percentage: {svmem.percent}%")

        if svmem.percent > 90:
            logging.warning("High Memory Usage Detected!")

    except Exception as e:
        logging.error(f"Failed to retrieve system info: {e}")