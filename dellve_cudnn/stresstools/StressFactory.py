'''
.. module:: stress_factory
    :platform: Unix
    :synopsis: Factory Stress Test Class

.. moduleauthor:: DELLveTeam

'''
from abc import ABCMeta, abstractmethod
import json
import re
import time

from dellve import Benchmark, BenchmarkInterrupt
import dellve_cudnn_benchmark as dcb
from helper import gpu_info

class StressToolFactory(Benchmark):
    '''Abstract factory class used by all stress tests in Dellve CuDNN.

    Child classes simply extend this function and define:
    :meth:`dellve_cudnn.stresstools.StressFactory.StressFactory.get_controller`
    '''
    __metaclass__ = ABCMeta

    config = {
        'gpu': '',
        'mem_util': 50,
        'seconds': 20,
    }
    '''Defines configuration defaults for dellve.

    Parameters:
        gpu: GPU to run benchmark.
        mem_util: Memory % to use for each stress test.
        seconds: Number of seconds to run the stress test for.
    '''

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
    def get_controller(self): 
        '''See:

        :meth:`dellve_cudnn.benchmarks.BenchmarkFactory.BenchmarkFactory.get_controller`
        '''
        pass

    def routine(self):
        '''Extension of dellve routine which specifies the function of
        each tool run when executed through the main program.

        Set up configuration required to run each test using the mem_util 
        provided. Then for each problem set, start the stress test 
        by passing in the gpu_id and number of seconds. 

        Periodically profile the stress tool to check progress and update
        the progress to the main program.
        '''
        config = self.__get_config()
        self.mem_util = config['mem_util']
        self.gpu_id = config['gpu_id']

        try:
            self.controller = self.get_controller()
            self.controller.start_stress_tool(config['gpu_id'], config['seconds'])

            while (not self.__complete()):
                self.__update_progress()
                time.sleep(0.1)

            self.__update_progress()
            # TODO: output performance details

        except BenchmarkInterrupt:
            print '\nCurrent stress tool has stopped'

    def __complete(self):
        return self.controller.get_progress() == 1.0

    def __update_progress(self):
        p = self.controller.get_progress()
        if (p > 0):
            self.progress = p * 100

    def __get_config(self):
        config = {}

        pattern = '(?:Device\s+)(\d+)'
        match = re.match(pattern, self.config['gpu'])
        config['gpu_id'] = int(match.group(1))

        config['mem_util'] = float(self.config['mem_util']) / 100
        config['seconds'] = self.config['seconds']

        return config
