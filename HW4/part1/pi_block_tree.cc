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
    int destination = 0;
    int tag = 0;
    MPI_Status status;
    long long int local_count = 0;
    unsigned seed;
    srand(seed);
    for (int i = 0; i < tosses / world_size; i++){
        double x = (double) rand_r(&seed) / (double)RAND_MAX;
        double y = (double) rand_r(&seed) / (double)RAND_MAX;
        if (x * x + y * y <= 1){
            local_count++;
        }
    }

    // TODO: binary tree redunction
    int difference = 1;
    int div = 2;
    long long int buf;
    while(difference < world_size){
        if(difference == 1){
            if(world_rank % div == 0){
                int source = world_rank + difference;
                MPI_Recv(&buf, 1, MPI_UNSIGNED_LONG, source, tag, MPI_COMM_WORLD, &status);
                local_count += buf;
            }
            else{
                int destination = world_rank - difference;
                // printf("world_rank:%d , diff: %d, des: %d\n", world_rank, difference, destination);
                MPI_Send(&local_count, 1, MPI_UNSIGNED_LONG, destination, tag, MPI_COMM_WORLD);
            }
        }
        else{
            if(!(world_rank % difference)){
                if((world_rank / difference) % 2 == 0){
                    int source = world_rank + difference;
                    MPI_Recv(&buf, 1, MPI_UNSIGNED_LONG, source, tag, MPI_COMM_WORLD, &status);
                    local_count += buf;
                }
                if((world_rank / difference) % 2 != 0){
                    int destination = world_rank - difference;
                    MPI_Send(&local_count, 1, MPI_UNSIGNED_LONG, destination, tag, MPI_COMM_WORLD);
                }
            }
        }
        difference *= 2;
    }

    if (world_rank == 0)
    {
        // TODO: PI result
        // print(total_cnt);
        pi_result = 4 * local_count / (double) tosses;

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}
