#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>

int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---

    // TODO: MPI init
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    unsigned total_cnt = 0;
    int destination = 0;
    int tag = 0;
    MPI_Status status;
    int local_count = 0;
    unsigned int seed;
    srand(seed);
    for (int i = 0; i < tosses / world_size; i++){
        double x = (double) rand_r(&seed) / (double)RAND_MAX;
        double y = (double) rand_r(&seed) / (double)RAND_MAX;
        if (x * x + y * y <= 1){
            local_count++;
        }
    }
    total_cnt += local_count;


    if (world_rank > 0)
    {
        // TODO: MPI workers
        MPI_Request req;
	    MPI_Isend(&local_count, 1, MPI_INT, destination, tag, MPI_COMM_WORLD, &req);

    }
    else if (world_rank == 0)
    {
        // TODO: non-blocking MPI communication.
        // Use MPI_Irecv, MPI_Wait or MPI_Waitall.
        MPI_Status *status = new MPI_Status[world_size - 1];
        MPI_Request *requests = new MPI_Request[world_size - 1];
        int buf[world_size - 1];
        for (int src = 1; src < world_size; src++){
            MPI_Irecv(&buf[src-1], 1, MPI_INT, src, tag, MPI_COMM_WORLD, &requests[src-1]);
        }

        MPI_Waitall(world_size - 1, requests, status);
        for(int i = 0; i < world_size - 1; i++){
            total_cnt += buf[i];
        }
        delete [] requests;
        delete [] status;

    }

    if (world_rank == 0)
    {
        // TODO: PI result
        pi_result = 4 * total_cnt / (double) tosses;

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}
