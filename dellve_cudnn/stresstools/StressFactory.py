from abc import ABCMeta, abstractmethod
import json
import re
import time

from dellve import Benchmark, BenchmarkInterrupt
import dellve_cudnn_benchmark as dcb
from helper import gpu_info

class StressToolFactory(Benchmark):
    __metaclass__ = ABCMeta
    config = {
        'gpu': '',
        'mem_util': 50,
        'seconds': 20,
    }

    schema = {
        'type': 'object',
        'properties': {
            'gpu': {
                'description': 'Device ID and name of GPU to run benchmark on',
                'type': 'string',
                'enum': [''],
            },
            'mem_util': {
                'description': 'GPU memory utilization in %',
                'type': 'integer',
                'minimum': 5,
                'maximum': 95,
            },
            'seconds': {
                'description': 'Number of seconds to run stress tool',
                'type': 'integer',
                'minimum': 5,
                'maximum': 100000,
            },
        },
        'required': ['gpu', 'mem_util', 'seconds'],
    }


    @abstractmethod
    def get_controller(self): pass

    def routine(self):
        config = self.get_config()
        self.mem_util = config['mem_util']
        self.gpu_id = config['gpu_id']

        try:
            self.controller = self.get_controller()
            self.controller.start_stress_tool(config['gpu_id'], config['seconds'])

            while (not self.complete()):
                self.update_progress()
                time.sleep(0.1)

            self.update_progress()
            # TODO: output performance details

        except BenchmarkInterrupt:
            print '\nCurrent stress tool has stopped'

    def complete(self):
        return self.controller.get_progress() == 1.0

    def update_progress(self):
        p = self.controller.get_progress()
        if (p > 0):
            self.progress = p * 100

    @classmethod
    def init_config(cls):
        available_gpus = gpu_info.get_valid_gpus()
        cls.config['gpu'] = available_gpus[0]
        cls.schema['properties']['gpu']['enum'] = available_gpus

    def get_config(self):
        config = {}

        pattern = '(?:Device\s+)(\d+)'
        match = re.match(pattern, self.config['gpu'])
        config['gpu_id'] = int(match.group(1))

        config['mem_util'] = float(self.config['mem_util']) / 100
        config['seconds'] = self.config['seconds']

        return config
