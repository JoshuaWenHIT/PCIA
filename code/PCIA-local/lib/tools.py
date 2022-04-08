import GPUtil


def get_gpu_status():
    count = 0
    status = []
    gpu_list = GPUtil.getGPUs()
    for gpu in gpu_list:
        gpu_dict = {}
        gpu_dict["gpuId"] = str(count)
        gpu_dict["Util"] = "{:.1f}%".format(gpu.load * 100)
        gpu_dict["Mem"] = "{:.1f}%".format(gpu.memoryUtil * 100)
        status.append(gpu_dict)
        count += 1
    return status
