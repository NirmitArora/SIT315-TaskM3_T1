#include <iostream>

#include <fstream>

#include <cstdlib>

#include <ctime>

#include <chrono>

#include <mpi.h>

#include <CL/cl.h>



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



// Function to perform local matrix multiplication with OpenCL

void multiplyMatricesCL(cl_context context, cl_device_id device, cl_command_queue queue, cl_program program, int A[N][N], int B[N][N], int C[N][N], int startRow, int endRow) {

    // Create buffer memory objects

    cl_mem memobjs[3];

    memobjs[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * N * N, A, NULL);

    memobjs[1] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * N * N, B, NULL);

    memobjs[2] = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * N * N, NULL, NULL);



    // Create the kernel

    cl_kernel kernel = clCreateKernel(program, "matrixMultiply", NULL);



    // Set the kernel arguments

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &memobjs[0]);

    clSetKernelArg(kernel, 1, sizeof(cl_mem), &memobjs[1]);

    clSetKernelArg(kernel, 2, sizeof(cl_mem), &memobjs[2]);

    clSetKernelArg(kernel, 3, sizeof(int), &N);



    // Set work-item dimensions

    size_t global_work_size[2] = { N, (size_t)(endRow - startRow) };



    // Execute kernel

    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);



    // Read output array

    clEnqueueReadBuffer(queue, memobjs[2], CL_TRUE, 0, sizeof(int) * N * (endRow - startRow), &C[startRow][0], 0, NULL, NULL);



    // Release memory objects and kernel

    for (int i = 0; i < 3; ++i) {

        clReleaseMemObject(memobjs[i]);

    }

    clReleaseKernel(kernel);

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



    // Scatter matrix A to all processes

    int* sendBuffer = new int[rowsPerProcess * N];

    MPI_Scatter(A, rowsPerProcess * N, MPI_INT, sendBuffer, rowsPerProcess * N, MPI_INT, 0, MPI_COMM_WORLD);

    int (*sendBuffer2D)[N] = reinterpret_cast<int (*)[N]>(sendBuffer);



    // Initialize OpenCL

    cl_platform_id platform;

    clGetPlatformIDs(1, &platform, NULL);

    cl_device_id device;

    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);

    cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);

    const char* source = "__kernel void matrixMultiply(__global int* A, __global int* B, __global int* C, int N) {"

                         "    int row = get_global_id(0);"

                         "    int col = get_global_id(1);"

                         "    int sum = 0;"

                         "    for (int i = 0; i < N; ++i) {"

                         "        sum += A[row * N + i] * B[i * N + col];"

                         "    }"

                         "    C[row * N + col] = sum;"

                         "}";

    cl_program program = clCreateProgramWithSource(context, 1, &source, NULL, NULL);

    clBuildProgram(program, 1, &device, NULL, NULL, NULL);



    // Record start time

    auto startTime = high_resolution_clock::now();



    // Perform local matrix multiplication using OpenCL

    multiplyMatricesCL(context, device, queue, program, sendBuffer2D, B, C, rank * rowsPerProcess, (rank + 1) * rowsPerProcess);



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



    // Release OpenCL resources

    clReleaseProgram(program);

    clReleaseCommandQueue(queue);

    clReleaseContext(context);



    // Finalize MPI

    MPI_Finalize();



    return 0;

}

