#include <mpi.h>
#include <iostream>
#include <fstream>
#include <ctime>
#include <cstdlib>

void multiply(int n, int A[1000][1000], int B[1000][1000], int C[1000][1000])
{
    int i, j, k;
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++) {
            C[i][j] = 0;
            for (k = 0; k < n; k++)
                C[i][j] += A[i][k] * B[k][j];
        }
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n;

    if (rank == 0) {
        std::cout << "Enter matrix size: ";
        std::cin >> n;
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    static int A[2000][2000];
    static int B[2000][2000];
    static int C[2000][2000];

    std::ofstream Afile("A_matrix.txt");
    std::ofstream Bfile("B_matrix.txt");

    srand(time(0) + rank);

    if (rank == 0) {

        std::ofstream Afile("A_matrix.txt");
        std::ofstream Bfile("B_matrix.txt");

        Afile << n << "\n";
        Bfile << n << "\n";

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                A[i][j] = rand() % 10;
                B[i][j] = rand() % 10;

                Afile << A[i][j] << " ";
                Bfile << B[i][j] << " ";
            }
            Afile << "\n";
            Bfile << "\n";
        }

        Afile.close();
        Bfile.close();
    }

    MPI_Bcast(B, 2000 * 2000, MPI_INT, 0, MPI_COMM_WORLD);

    int rows = n / size;

    static int local_A[2000][2000];
    static int local_C[2000][2000];

    MPI_Scatter(A, rows * 2000, MPI_INT,
        local_A, rows * 2000, MPI_INT,
        0, MPI_COMM_WORLD);

    double start = MPI_Wtime();


    for (int i = 0; i < rows; i++)
        for (int j = 0; j < n; j++) {
            local_C[i][j] = 0;
            for (int k = 0; k < n; k++)
                local_C[i][j] += local_A[i][k] * B[k][j];
        }

    double end = MPI_Wtime();


    MPI_Gather(local_C, rows * 2000, MPI_INT,
        C, rows * 2000, MPI_INT,
        0, MPI_COMM_WORLD);


    if (rank == 0) {

        std::ofstream Cfile("Result.txt");

        Cfile << n << "\n";
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++)
                Cfile << C[i][j] << " ";
            Cfile << "\n";
        }

        Cfile.close();

        std::cout << "\n======== RESULT ========\n";
        std::cout << "Matrix size: " << n << " x " << n << "\n";
        std::cout << "Processes: " << size << "\n";
        std::cout << "Time: " << (end - start) << " sec\n";
        std::cout << "Operations: " << (long long)n * n * n << "\n";
    }

    MPI_Finalize();
    return 0;
}