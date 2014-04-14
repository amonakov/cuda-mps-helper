cuda-mps-helper
===============

This script can be used to automate starting and stopping CUDA MPS servers on
multi-GPU nodes, and running MPI processes with MPS dispatch.

To start:

    ./manage-mps.sh start

On a system with both Tesla and GeForce-class cards installed, some GeForce
cards may be enumerated in CUDA_VISIBLE_DEVICES before Teslas.  In that case,
run `deviceQuery` CUDA sample to find out the order and pass the device
numbers as the argument to 'start' command:

    ./manage-mps.sh start 0 2

To signal MPS servers to shut down and remove temporary directories:

    ./manage-mps.sh stop

To run a process via one of MPS server, distributing in a round-robin fashion
by MPI local rank:

    ./manage-mps.sh wrap *COMMAND*

For instance try

    mpirun -np 4 ./manage-mps.sh wrap bash -c 'echo $CUDA_MPS_PIPE_DIRECTORY'
