
from pynvml import *
# import dellve_cudnn_benchmark as dcb

def get_names():
    names = []

    nvmlInit()
    device_count = nvmlDeviceGetCount()
    for i in range(device_count):
        #TODO: implement a helper function in dcb that checks 
        #      if GPU architecture is supported
        #if dcb.valid_gpu(i):
        handle = nvmlDeviceGetHandleByIndex(i)
        names.append('Device {} : {}'.format(i, nvmlDeviceGetName(handle)))
    nvmlShutdown()

    return names

def get_total_mem(device_id):
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(device_id)
    mem = nvmlDeviceGetMemoryInfo(handle).total
    nvmlShutdown()

    return mem
