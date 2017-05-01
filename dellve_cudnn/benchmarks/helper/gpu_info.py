import multiprocessing as mp
from pynvml import *
import dellve_cudnn_helper as dch

# Note:
#    We use multiprocessing to spawn a new process for all CUDA/CuDNN-related code
#    to ensure that we free GPU resources once we're done
def get_valid_gpus_target(name_queue):
    nvmlInit()
    device_count = nvmlDeviceGetCount()

    for i in range(device_count + 1):
        if dch.is_valid_gpu(i):
            handle = nvmlDeviceGetHandleByIndex(i)
            name_queue.put('Device {} : {}'.format(i, nvmlDeviceGetName(handle)))

    nvmlShutdown()

def get_valid_gpus():
    gpu_names_queue = mp.Queue()
    p = mp.Process(target=get_valid_gpus_target, args=(gpu_names_queue,))
    p.start()
    p.join()

    gpu_names_list = []
    while gpu_names_queue.qsize() != 0:
        gpu_names_list.append(gpu_names_queue.get())

    return gpu_names_list
