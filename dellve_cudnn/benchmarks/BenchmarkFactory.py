from abc import ABCMeta, abstractmethod
import json
import re
import time

from dellve import Benchmark, BenchmarkInterrupt
from helper import gpu_info

class BenchmarkFactory(Benchmark):
    __metaclass__ = ABCMeta
    config = {
        'gpu': '',
        'num_runs': 50,
    }

    schema = {
        'type': 'object',
        'properties': {
            'gpu': {
                'description': 'Device ID and name of GPU to run benchmark on',
                'type': 'string',
                'enum': ['']
            },
            'num_runs': {
                'description': 'Number of times to run operation per problem size',
                'type': 'integer',
                'minimum': 25,
                'maximum': 100,
            },
        },
        'required': ['gpu', 'num_runs'],
    }

    @abstractmethod
    def get_controller(self): pass

    @abstractmethod
    def get_problem_set(self): pass

    @abstractmethod
    def get_problem_header(self): pass

    def routine(self):
        config = self.get_config()

        control_constructor = self.get_controller()
        problem_set = self.get_problem_set()
        problem_set_size = len(problem_set)

        self.print_header(problem_set)

        for problem_number, problem in enumerate(problem_set):
            self.controller = control_constructor(*problem)

            try:
                self.controller.start_benchmark(config['gpu_id'], config['num_runs'])

                while (not self.complete()):
                    self.update_progress(problem_number, problem_set_size)
                    time.sleep(0.25)

                self.update_progress(problem_number, problem_set_size)
                self.generate_report(problem_set, problem_number, 
                                     self.controller.get_avg_time_micro())

            except BenchmarkInterrupt:
                print '\nCurrent benchmark has stopped'
                break


    def complete(self):
        return self.controller.get_progress() == 1.0

    def update_progress(self, problem_number, problem_set_size):
        p = self.controller.get_progress()
        if (p > 0):
            self.progress = (problem_number * 100. / problem_set_size) \
                          + (p * 100 / problem_set_size)

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

        config['num_runs'] = self.config['num_runs']

        return config

    def print_header(self, problem_set):
        row_format = '{:>2}' + '{:>9}' * len(problem_set[0]) + '{:>10}'

        header = ['#'] + self.get_problem_header() + ['time (us)']
        print row_format.format(*header)

    def generate_report(self, problem_set, problem_number, result):
        row_format = '{:>2}' + '{:>9}' * len(problem_set[0]) + '{:>10}'

        row = [problem_number] + problem_set[problem_number] + [result]
        print row_format.format(*row)

