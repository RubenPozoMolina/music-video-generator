import gc
import logging

import torch


class GPUUtils:

    @staticmethod
    def show_gpu_info():
        gpus = torch.cuda.device_count()
        for i in range(gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"Memory Usage: {torch.cuda.memory_allocated(i) / (1024**3):.2f} GB")
            print(f"Memory Capacity: {torch.cuda.get_device_properties(i).total_memory / (1024**3):.2f} GB")
            print(f"Memory Utilization: {(torch.cuda.memory_allocated(i) / torch.cuda.get_device_properties(i).total_memory) * 100:.2f}%")
            print("-" * 40)

    @staticmethod
    def show_mem():
        gpus = torch.cuda.device_count()
        for i in range(gpus):
            mem_usage = f"Memory Usage: {torch.cuda.memory_allocated(i) / (1024**3):.2f} GB "
            mem_capacity = f"Memory Capacity: {torch.cuda.get_device_properties(i).total_memory / (1024**3):.2f} GB "
            mem_utilization = f"Memory Utilization: {(torch.cuda.memory_allocated(i) / torch.cuda.get_device_properties(i).total_memory) * 100:.2f}%"
            logging.info(
                "%s %s %s",
                mem_usage,
                mem_capacity,
                mem_utilization
            )

    @staticmethod
    def free_memory():
        torch.cuda.empty_cache()
        gc.collect()
        print("Memory freed.")