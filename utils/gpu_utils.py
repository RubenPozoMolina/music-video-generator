import gc
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
    def get_gpu_memory_usage():
        gpus = torch.cuda.device_count()
        print("GPUS", gpus)
        for i in range(gpus):
            allocated = torch.cuda.memory_allocated(i-1) / (1024**3)
            total = torch.cuda.get_device_properties(i-1).total_memory / (1024**3)
            percentage = (allocated / total) * 100
            print(f"GPU {i-1} Memory Usage: {allocated:.2f} GB / {total:.2f} GB ({percentage:.2f}%)")

    @staticmethod
    def free_memory():
        torch.cuda.empty_cache()
        gc.collect()
        print("Memory freed.")