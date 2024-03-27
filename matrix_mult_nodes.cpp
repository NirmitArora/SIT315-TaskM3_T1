#include <iostream>

#include <fstream>

#include <cstdlib>

#include <ctime>

#include <chrono>

#include <mpi.h>



using namespace std;

using namespace std::chrono;



const int N = 400; // Define matrix size



// Function to initialize a matrix with random values

void initializeMatrix(int matrix[N][N]) {

    // Initialize matrix with random values

    for (int i = 0; i < N; ++i) {

        for (int j = 0; j < N; ++j) {

            matrix[i][j] = rand() % 10; // Generate random value between 0 and 9

        }

    }

}



// Function to perform local matrix multiplication

void multiplyMatrices(int A[N][N], int B[N][N], int C[N][N], int startRow, int endRow) {

    // Perform matrix multiplication for rows from startRow to endRow

    for (int i = startRow; i < endRow; ++i) {

        for (int j = 0; j < N; ++j) {

            C[i][j] = 0; // Initialize element of C to 0

            for (int k = 0; k < N; ++k) {

                C[i][j] += A[i][k] * B[k][j]; // Compute sum of products

            }

        }

    }

}



int main(int argc, char* argv[]) {

    // Initialize MPI

    MPI_Init(&argc, &argv);



    // Get the rank and size of the MPI communicator

    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Comm_size(MPI_COMM_WORLD, &size);



    // Seed random number generator

    srand(time(0));



    // Declare matrices A, B, and C

    int A[N][N];

    int B[N][N];

    int C[N][N];



    // Master process initializes matrices A and B

    if (rank == 0) {

        initializeMatrix(A);

        initializeMatrix(B);

    }



    // Broadcast matrix B to all processes

    MPI_Bcast(B, N*N, MPI_INT, 0, MPI_COMM_WORLD);



    // Define number of rows per process

    int rowsPerProcess = N / size;



    // Allocate separate buffer for sending data

    int* sendBuffer = new int[rowsPerProcess * N];



    // Scatter matrix A to all processes

    MPI_Scatter(A, rowsPerProcess * N, MPI_INT, sendBuffer, rowsPerProcess * N, MPI_INT, 0, MPI_COMM_WORLD);



    // Reinterpret sendBuffer as a two-dimensional array

    int (*sendBuffer2D)[N] = reinterpret_cast<int (*)[N]>(sendBuffer);



    // Record start time

    auto startTime = high_resolution_clock::now();



    // Perform local matrix multiplication

    multiplyMatrices(sendBuffer2D, B, C, rank * rowsPerProcess, (rank + 1) * rowsPerProcess);



    // Record end time

    auto endTime = high_resolution_clock::now();



    // Calculate elapsed time

    auto duration = duration_cast<microseconds>(endTime - startTime);



    // Print elapsed time on the master process

    if (rank == 0) {

        cout << "Time taken for matrix multiplication: " << duration.count() << " microseconds" << endl;

    }



    // Deallocate memory for the send buffer

    delete[] sendBuffer;



    // Finalize MPI

    MPI_Finalize();



    return 0;

}

