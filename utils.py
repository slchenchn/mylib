'''
Author: Shuailin Chen
Created Date: 2021-10-23
Last Modified: 2021-10-24
	content: 
'''

from gpustat import GPUStatCollection
import time
import argparse
import numpy as np
from numpy import ndarray


def wait_for_gpu(required_GB, interval=10):
    ''' Wait for available gpu, that is, exit until free gpu memory is greater than required.

    Args:
        required (float): required gpu memory in GB. None means no wait,
            negative values mean maximum requirment.
        interval (float): repeated interval in seconds. Default: 10
    '''

    if required_GB is None:
        return
    else:
        required_MB = required_GB * 1024

    print(f'waiting for availabel gpu memory: {required_MB} MB ...')
    while True:
        gpu_stats = GPUStatCollection.new_query()
        mem_free = [gpu.memory_free for gpu in gpu_stats.gpus]

        if (not isinstance(required_MB, ndarray)) and required_MB < 0:
            ''' maximum requriement, perseved 400MB for root or xorg'''
            required_MB = gpu_stats.gpus[0].memory_total-400

        if not isinstance(required_MB, (tuple, list, ndarray)):
            required_MB = [required_MB] * len(mem_free)

        mem_free = np.array(mem_free)
        required_MB = np.array(required_MB)

        if np.all(mem_free > required_MB):
            return
        else:
            time.sleep(interval)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--required', type=float, default=None)
    argparser.add_argument('--interval', type=float, default=10)

    args = argparser.parse_args()
    wait_for_gpu(args.required, args.interval)
