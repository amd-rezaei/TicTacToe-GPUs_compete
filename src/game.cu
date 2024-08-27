#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define ROWS 6
#define COLUMNS 7

// Kernel for Player 1: Random Strategy
__global__ void randomMove(int *board, int player)
{
    int col = threadIdx.x % COLUMNS; // Safe guarding against out-of-bounds
    if (board[col * ROWS] == 0)
    {
        for (int i = ROWS - 1; i >= 0; i--)
        {
            if (board[col * ROWS + i] == 0)
            {
                board[col * ROWS + i] = player;
                break;
            }
        }
    }
}

// Kernel for Player 2: Lookahead Strategy
__global__ void lookaheadMove(int *board, int player)
{
    int col = threadIdx.x % COLUMNS;
    // Add more sophisticated logic for lookahead here
}

// Check for a win condition
bool checkWin(int *board)
{
    // Check horizontally, vertically, and diagonally
    return false; // Add real check logic here
}

// Display the board state
void printBoard(int *board)
{
    for (int i = 0; i < ROWS; i++)
    {
        for (int j = 0; j < COLUMNS; j++)
        {
            int token = board[j * ROWS + i];
            printf("%c ", (token == 0) ? '.' : (token == 1 ? 'X' : 'O'));
        }
        printf("\n");
    }
}

int main()
{
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    int device_id = 0; // Default to the first GPU

    if (nDevices > 1)
    {
        // If more than one GPU, select a random one
        srand(time(NULL));
        device_id = rand() % nDevices;
    }
    cudaSetDevice(device_id);
    printf("Using GPU %d\n", device_id);

    int *board;
    cudaMallocManaged(&board, ROWS * COLUMNS * sizeof(int));
    memset(board, 0, ROWS * COLUMNS * sizeof(int));

    int currentPlayer = 1;
    while (!checkWin(board))
    {
        if (currentPlayer == 1)
        {
            randomMove<<<1, COLUMNS>>>(board, currentPlayer);
        }
        else
        {
            lookaheadMove<<<1, COLUMNS>>>(board, currentPlayer);
        }
        cudaDeviceSynchronize();
        printBoard(board);
        currentPlayer = 3 - currentPlayer; // Toggle between player 1 and 2
    }

    cudaFree(board);
    return 0;
}

