#include <string>
#include <stdio.h>

#include "cudnn_activation_driver.hpp"
#include "cudnn_problem_set.hpp"
#include "cli_parser.hpp"

int main(int argc, char *argv[]) {
    CLIParser options(argc, argv); 
    CudnnActivationProblemSet problems(options.getProblemSetFile());
    CudnnActivationDriver driver(CudnnActivationMethod::FORWARD, problems, options.getNumRuns(),
                            options.getGpus());

    printf("Starting Forward Activation through %s set with %d runs\n", options.getProblemSetFile().c_str(), 
            options.getNumRuns());
    for (auto i = 0; i < problems.getSize(); i++) {
        int time = driver.run(i);
        printf("Set %d took avg %d us\n", i+1, time);
    }
    
    return 0;
}
