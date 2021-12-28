'''
Author: Shuailin Chen
Created Date: 2021-10-23
Last Modified: 2021-12-28
	content: 
'''

import os
import sys
import time
import argparse
import numpy as np
from numpy import ndarray

from gpustat import GPUStatCollection
from blessings import Terminal
import locale


class MyGPUStatCollection(GPUStatCollection):

    def __init__(self,
                gpu_stat_collection=None,
                gpu_list=None,
                driver_version=None):

        if gpu_stat_collection is not None:
            self.gpus = gpu_stat_collection.gpus
            self.hostname = gpu_stat_collection.hostname
            self.query_time = gpu_stat_collection.query_time
            self.driver_version = gpu_stat_collection.driver_version
        else:
            super().__init__(gpu_list, driver_version=driver_version)

    @staticmethod
    def new_query():
        return MyGPUStatCollection(GPUStatCollection.new_query())

    def log_gpu_info_json(self, logger):
        def date_handler(obj):
            if hasattr(obj, 'isoformat'):
                return obj.isoformat()
            else:
                raise TypeError(type(obj))

        o = self.jsonify()
        logger.info(o)

    def log_gpu_info(self,
                    logger,
                    force_color=False,
                    no_color=False,
                    show_cmd=False,
                    show_user=False,
                    show_pid=False,
                    show_power=None,
                    show_fan_speed=None,
                    gpuname_width=16,
                    show_header=True,
                    eol_char=os.linesep,
                    ):
            # ANSI color configuration
            if force_color and no_color:
                raise ValueError("--color and --no_color can't"
                                " be used at the same time")

            if force_color:
                t_color = Terminal(kind='linux', force_styling=True)

                # workaround of issue #32 (watch doesn't recognize sgr0 characters)
                t_color.normal = u'\x1b[0;10m'
            elif no_color:
                t_color = Terminal(force_styling=None)
            else:
                t_color = Terminal()   # auto, depending on isatty

            # appearance settings
            entry_name_width = [len(g.entry['name']) for g in self]
            gpuname_width = max([gpuname_width or 0] + entry_name_width)

            # header
            if show_header:
                time_format = locale.nl_langinfo(locale.D_T_FMT)

                header_template = '{t.bold_white}{hostname:{width}}{t.normal}  '
                header_template += '{timestr}  '
                header_template += '{t.bold_black}{driver_version}{t.normal}'

                header_msg = header_template.format(
                        hostname=self.hostname,
                        width=gpuname_width + 3,  # len("[?]")
                        timestr=self.query_time.strftime(time_format),
                        driver_version=self.driver_version,
                        t=t_color,
                    )

                logger.info(header_msg.strip())
                logger.info(eol_char)

            # body
            for g in self:
                g.print_to(fp,
                        show_cmd=show_cmd,
                        show_user=show_user,
                        show_pid=show_pid,
                        show_power=show_power,
                        show_fan_speed=show_fan_speed,
                        gpuname_width=gpuname_width,
                        term=t_color)
                logger.info(eol_char)


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
