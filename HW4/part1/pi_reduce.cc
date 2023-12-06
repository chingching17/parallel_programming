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

    long long int total_cnt = 0;
    int buf = 0;
    int destination = 0;
    int tag = 0;
    MPI_Status status;
    long long int local_count = 0;
    unsigned int seed;
    srand(seed);
    for (int i = 0; i < tosses / world_size; i++){
        double x = (double) rand_r(&seed) / (double)RAND_MAX;
        double y = (double) rand_r(&seed) / (double)RAND_MAX;
        if (x * x + y * y <= 1){
            local_count++;
        }
    }

    // TODO: use MPI_Reduce
    MPI_Reduce(&local_count, &total_cnt, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

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
