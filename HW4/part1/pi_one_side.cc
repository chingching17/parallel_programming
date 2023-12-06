#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>

bool done_yet(long long int *recvBuf, int size){
    int done = 1;
    for (int i = 1; i < size; i++){
        if(recvBuf[i] == 0){
            done = 0;
            break;
        }
    }
    return done;
}

int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---

    MPI_Win win;

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

    if (world_rank == 0)
    {
        // Master
        long long int  *recvArray;
        MPI_Alloc_mem(world_size * sizeof(long long int), MPI_INFO_NULL, &recvArray);

        for (int i = 0; i < world_size; i++)
        {
           recvArray[i] = 0;
        }

        MPI_Win_create(recvArray, world_size * sizeof(long long int), sizeof(long long int), MPI_INFO_NULL,
           MPI_COMM_WORLD, &win);
        
        bool done = 0;
        while(!done){
            MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, win);
            done = done_yet(recvArray, world_size);
            MPI_Win_unlock(0, win);
        }

        total_cnt += local_count;
        for (int i = 1; i < world_size; i++){
            total_cnt += recvArray[i];
        }

        MPI_Win_free(&win);
        MPI_Free_mem(recvArray);
    }
    else
    {
        MPI_Win_create(NULL, 0, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &win);

        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win);
        MPI_Put(&local_count, 1, MPI_UNSIGNED_LONG, 0, world_rank, 1, MPI_UNSIGNED_LONG, win);
        MPI_Win_unlock(0, win);

        MPI_Win_free(&win);
    }

    if (world_rank == 0)
    {
        // TODO: handle PI result
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