#define _GNU_SOURCE
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    char hostname[256];
    gethostname(hostname, sizeof(hostname));

    printf("[Rank %d of %d] Initialized on host: %s\n", world_rank, world_size, hostname);
    MPI_Barrier(MPI_COMM_WORLD);

    int token;
    double start_time = MPI_Wtime();

    if (world_rank == 0) {
        token = 42;
        printf("\n>>> [Rank 0] Originating token %d to Rank 1...\n", token);
        MPI_Send(&token, 1, MPI_INT, (world_rank + 1) % world_size, 0, MPI_COMM_WORLD);
        MPI_Recv(&token, 1, MPI_INT, world_size - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf(">>> [Rank 0] Successfully received token %d back from Rank %d!\n", token, world_size - 1);
    } else {
        MPI_Recv(&token, 1, MPI_INT, world_rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("    [Rank %d on %s] Received token %d, forwarding to Rank %d...\n", 
               world_rank, hostname, token, (world_rank + 1) % world_size);
        MPI_Send(&token, 1, MPI_INT, (world_rank + 1) % world_size, 0, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();

    if (world_rank == 0) {
        printf("\nRing pass completed in %f milliseconds across %d ranks.\n\n", 
               (end_time - start_time) * 1000.0, world_size);
    }

    MPI_Finalize();
    return 0;
}
