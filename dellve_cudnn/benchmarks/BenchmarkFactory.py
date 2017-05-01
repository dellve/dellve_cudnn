'''
.. module:: benchmark_factory
    :platform: Unix
    :synopsis: Factory Benchmark Class

.. moduleauthor:: DELLveTeam

'''
from abc import ABCMeta, abstractmethod
import json
import re
import time

from dellve import Benchmark, BenchmarkInterrupt
from helper import gpu_info

class BenchmarkFactory(Benchmark):
    '''Abstract factory class used by all benchmarks in Dellve CuDNN.

    Child classes simply extend this function and define:
    :meth:`dellve_cudnn.benchmarks.BenchmarkFactory.BenchmarkFactory.get_controller`
    :meth:`dellve_cudnn.benchmarks.BenchmarkFactory.BenchmarkFactory.get_problem_set`
    :meth:`dellve_cudnn.benchmarks.BenchmarkFactory.BenchmarkFactory.get_problem_header`
    '''
    __metaclass__ = ABCMeta

    config = {
        'gpu': '',
        'num_runs': 50,
    }
    '''Defines configuration defaults for dellve.
    
    Parameters:
        gpu: GPU to run benchmark.
        num_runs: number of times to run benchmark and get average time.
    '''

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
    def get_controller(self): 
       '''Get the controller from Dellve CuDNN PyBind.

       These controllers are defined within cpp and run the stress tools
       across all of the operations, each which has its own method.

       Return:
            controller unique to each benchmark.
       '''

       pass

    @abstractmethod
    def get_problem_set(self): 
        '''Get the problem set through a csv file that has all 
        of the problem sets that this benchmark will test. 

        This csv file can be edited in folder:
        dellve_cudnn.benchmarks.problemsets

        Returns:
            Problem set unique to each benchmark.
        '''
        
        pass

    @abstractmethod
    def get_problem_header(self): 
        '''Get header of CSV file. Used to format the output into a nice
        table.

        Returns:
            Problem header unique to each problem set.
        '''
        pass

    def routine(self):
        '''Extension of dellve routine which specifies the function of
        each tool run when executed through the main program.

        Sets up the configuration required to run each test, then for
        each problem set, start the benchmark by passing in the gpu_id
        and num_runs. 
        
        Periodically profile the benchmark to check progress and update
        the progress to the main program. Once the test finishes,
        generate a report based on the average time taken across all
        of the num_runs.
        '''
        config = self.__get_config()

        control_constructor = self.get_controller()
        problem_set = self.get_problem_set()
        problem_set_size = len(problem_set)

        self.__print_header(problem_set)

        for problem_number, problem in enumerate(problem_set):
            self.controller = control_constructor(*problem)

            try:
                self.controller.start_benchmark(config['gpu_id'], config['num_runs'])

                while (not self.__complete()):
                    self.__update_progress(problem_number, problem_set_size)
                    time.sleep(0.25)

                self.__update_progress(problem_number, problem_set_size)
                self.__generate_report(problem_set, problem_number, 
                                     self.controller.get_avg_time_micro())

            except BenchmarkInterrupt:
                print '\nCurrent benchmark has stopped'
                break


    def __complete(self):
        return self.controller.get_progress() == 1.0

    def __update_progress(self, problem_number, problem_set_size):
        p = self.controller.get_progress()
        if (p > 0):
            self.progress = (problem_number * 100. / problem_set_size) \
                          + (p * 100 / problem_set_size)

    def __get_config(self):
        config = {}

        pattern = '(?:Device\s+)(\d+)'
        match = re.match(pattern, self.config['gpu'])
        config['gpu_id'] = int(match.group(1))

        config['num_runs'] = self.config['num_runs']

        return config

    def __print_header(self, problem_set):
        row_format = '{:>2}' + '{:>9}' * len(problem_set[0]) + '{:>10}'

        header = ['#'] + self.get_problem_header() + ['time (us)']
        print row_format.format(*header)

    def __generate_report(self, problem_set, problem_number, result):
        row_format = '{:>2}' + '{:>9}' * len(problem_set[0]) + '{:>10}'

        row = [problem_number] + problem_set[problem_number] + [result]
        print row_format.format(*row)

