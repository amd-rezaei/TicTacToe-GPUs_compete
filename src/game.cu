#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <time.h>

#define ROWS 6
#define COLUMNS 7

// Kernel for Player 1: Random Strategy
__global__ void randomMove(int *board, int player)
{
    int col = threadIdx.x % COLUMNS;
    if (board[col * ROWS] == 0)
    { // Check if the top of the column is empty
        for (int i = ROWS - 1; i >= 0; i--)
        {
            if (board[col * ROWS + i] == 0)
            {
                board[col * ROWS + i] = player;
                return;
            }
        }
    }
}

// Kernel for Player 2: Lookahead Strategy
__global__ void lookaheadMove(int *board, int player)
{
    // Placeholder for a more complex strategy.
    // No need to declare 'col' if it's not being used.
}

// Check for a win condition, returns true if player wins
bool checkWin(int *board, int player)
{
    // Horizontal check
    for (int row = 0; row < ROWS; row++)
    {
        for (int col = 0; col < COLUMNS - 3; col++)
        {
            if (board[col * ROWS + row] == player && board[(col + 1) * ROWS + row] == player &&
                board[(col + 2) * ROWS + row] == player && board[(col + 3) * ROWS + row] == player)
            {
                return true;
            }
        }
    }

    // Vertical check
    for (int col = 0; col < COLUMNS; col++)
    {
        for (int row = 0; row < ROWS - 3; row++)
        {
            if (board[col * ROWS + row] == player && board[col * ROWS + (row + 1)] == player &&
                board[col * ROWS + (row + 2)] == player && board[col * ROWS + (row + 3)] == player)
            {
                return true;
            }
        }
    }

    // Diagonal checks
    // Diagonal down-right
    for (int row = 0; row < ROWS - 3; row++)
    {
        for (int col = 0; col < COLUMNS - 3; col++)
        {
            if (board[col * ROWS + row] == player && board[(col + 1) * ROWS + (row + 1)] == player &&
                board[(col + 2) * ROWS + (row + 2)] == player && board[(col + 3) * ROWS + (row + 3)] == player)
            {
                return true;
            }
        }
    }

    // Diagonal down-left
    for (int row = 0; row < ROWS - 3; row++)
    {
        for (int col = 3; col < COLUMNS; col++)
        {
            if (board[col * ROWS + row] == player && board[(col - 1) * ROWS + (row + 1)] == player &&
                board[(col - 2) * ROWS + (row + 2)] == player && board[(col - 3) * ROWS + (row + 3)] == player)
            {
                return true;
            }
        }
    }

    return false;
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
    while (true)
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

        if (checkWin(board, currentPlayer))
        {
            printf("Player %d wins!\n", currentPlayer);
            break;
        }

        currentPlayer = 3 - currentPlayer; // Toggle between player 1 and 2
    }

    cudaFree(board);
    return 0;
}
