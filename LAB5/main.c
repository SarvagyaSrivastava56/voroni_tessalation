#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "perfect_worker.h"

#define MAX_VAL 10000 // Upper limit to test
#define TAG_DATA 1
#define TAG_STOP 2

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) printf("This pattern requires a Master and at least one Slave.\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        // --- MASTER LOGIC ---
        int current_num = 2;
        int active_workers = size - 1;
        int result_msg;
        MPI_Status status;
        
        // Timing variables
        double start_time = MPI_Wtime();
        double total_comm_time = 0.0;
        double comm_start, comm_end;
        int numbers_tested = 0;

        printf("Searching for Perfect Numbers up to %d...\n", MAX_VAL);

        while (active_workers > 0) {
            // a. Wait for a message from any slave
            comm_start = MPI_Wtime();
            MPI_Recv(&result_msg, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            comm_end = MPI_Wtime();
            total_comm_time += (comm_end - comm_start);
            
            int slave_rank = status.MPI_SOURCE;

            // b. Check if message is perfect, not perfect, or initial (0)
            if (result_msg > 0) {
                printf("Found Perfect Number: %d\n", result_msg);
            }
            if (result_msg != 0) {
                numbers_tested++;
            }
            
            // c. Send next number to test or stop signal
            comm_start = MPI_Wtime();
            if (current_num <= MAX_VAL) {
                MPI_Send(&current_num, 1, MPI_INT, slave_rank, TAG_DATA, MPI_COMM_WORLD);
                current_num++;
            } else {
                int stop_signal = 0;
                MPI_Send(&stop_signal, 1, MPI_INT, slave_rank, TAG_STOP, MPI_COMM_WORLD);
                active_workers--;
            }
            comm_end = MPI_Wtime();
            total_comm_time += (comm_end - comm_start);
        }
        
        // Print timing statistics
        double end_time = MPI_Wtime();
        double total_time = end_time - start_time;
        double computation_time = total_time - total_comm_time;
        
        printf("\n=== TIMING STATISTICS ===\n");
        printf("Total execution time: %.4f seconds\n", total_time);
        printf("Communication time: %.4f seconds\n", total_comm_time);
        printf("Computation time: %.4f seconds\n", computation_time);
        printf("Communication overhead: %.2f%%\n", (total_comm_time / total_time) * 100);
        printf("Numbers tested: %d\n", numbers_tested);
        printf("Average time per number: %.6f seconds\n", total_time / numbers_tested);
    } else {
        // --- SLAVE LOGIC ---
        int request_msg = 0; // Initial request is zero
        int number_to_test;
        MPI_Status status;
        
        // Timing variables for slaves
        double start_time = MPI_Wtime();
        double total_comm_time = 0.0;
        double total_comp_time = 0.0;
        double comm_start, comm_end, comp_start, comp_end;
        int numbers_processed = 0;

        while (1) {
            // Send result (or initial 0) to Master
            comm_start = MPI_Wtime();
            MPI_Send(&request_msg, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            comm_end = MPI_Wtime();
            total_comm_time += (comm_end - comm_start);
            
            // Receive task
            comm_start = MPI_Wtime();
            MPI_Recv(&number_to_test, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            comm_end = MPI_Wtime();
            total_comm_time += (comm_end - comm_start);
            
            if (status.MPI_TAG == TAG_STOP) break;

            // Test and prepare result for next send
            comp_start = MPI_Wtime();
            request_msg = test_perfect(number_to_test);
            comp_end = MPI_Wtime();
            total_comp_time += (comp_end - comp_start);
            numbers_processed++;
        }
        
        // Print slave timing statistics
        double end_time = MPI_Wtime();
        double total_time = end_time - start_time;
        
        printf("Slave %d: Total time: %.4fs, Comm: %.4fs (%.1f%%), Comp: %.4fs (%.1f%%), Numbers: %d\n", 
               rank, total_time, total_comm_time, 
               (total_comm_time/total_time)*100, total_comp_time, 
               (total_comp_time/total_time)*100, numbers_processed);
    }

    MPI_Finalize();
    return 0;
}