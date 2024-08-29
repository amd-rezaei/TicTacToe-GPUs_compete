#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <time.h>

// Default values
#define DEFAULT_ROWS 3
#define DEFAULT_COLUMNS 3

// Kernel for Player 1: Random Move Strategy
__global__ void randomMove(int *board, int player, int rows, int columns, unsigned long long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0)
    {
        curandState_t state;
        curand_init(seed, idx, 0, &state);

        int col = curand(&state) % columns;
        int startCol = col;

        while (true)
        {
            if (board[col * rows] == 0)
            {
                for (int i = rows - 1; i >= 0; i--)
                {
                    if (board[col * rows + i] == 0)
                    {
                        board[col * rows + i] = player;
                        return;
                    }
                }
            }
            col = (col + 1) % columns;
            if (col == startCol)
                break; // Break if we've checked all columns
        }
    }
}

// Kernel for Player 2: Lookahead Strategy
__global__ void lookaheadMove(int *board, int player, int rows, int columns)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0)
    {
        int opponent = 3 - player;

        for (int col = 0; col < columns; col++)
        {
            for (int i = rows - 1; i >= 0; i--)
            {
                if (board[col * rows + i] == 0)
                {
                    board[col * rows + i] = opponent;
                    bool win = false;

                    // Horizontal check
                    for (int row = 0; row < rows; row++)
                    {
                        if (board[row * columns] == opponent &&
                            board[row * columns + 1] == opponent &&
                            board[row * columns + 2] == opponent)
                        {
                            win = true;
                        }
                    }

                    // Vertical check
                    for (int colcnt = 0; colcnt < columns; colcnt++)
                    {
                        if (board[colcnt] == opponent &&
                            board[colcnt + columns] == opponent &&
                            board[colcnt + 2 * columns] == opponent)
                        {
                            win = true;
                        }
                    }

                    // Diagonal check (top-left to bottom-right)
                    if (board[0] == opponent &&
                        board[4] == opponent &&
                        board[8] == opponent)
                    {
                        win = true;
                    }

                    // Diagonal check (top-right to bottom-left)
                    if (board[2] == opponent &&
                        board[4] == opponent &&
                        board[6] == opponent)
                    {
                        win = true;
                    }

                    if (win)
                    {
                        board[col * rows + i] = player;
                        return;
                    }

                    board[col * rows + i] = 0;
                    break;
                }
            }
        }

        // If no winning move to block, make a random move
        for (int col = 0; col < columns; col++)
        {
            if (board[col * rows] == 0)
            {
                for (int i = rows - 1; i >= 0; i--)
                {
                    if (board[col * rows + i] == 0)
                    {
                        board[col * rows + i] = player;
                        return;
                    }
                }
            }
        }
    }
}

// Utility function to print the current board state
void printBoard(const int *board, int rows, int columns)
{
    printf("Current Board State:\n");
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < columns; ++j)
        {
            printf("%d ", board[i * columns + j]);
        }
        printf("\n");
    }
    printf("\n");
}

// Enhanced win check function
bool checkWin(const int *board, int player, int rows, int columns)
{
    int horizontalWin = rows;  // Win condition for horizontal and diagonal checks
    int verticalWin = columns; // Win condition for vertical checks

    // Horizontal check
    for (int row = 0; row < rows; row++)
    {
        for (int col = 0; col <= columns - horizontalWin; col++)
        {
            bool win = true;
            for (int k = 0; k < horizontalWin; k++)
            {
                if (board[row * columns + col + k] != player)
                {
                    win = false;
                    break;
                }
            }
            if (win)
                return true;
        }
    }

    // Vertical check
    for (int col = 0; col < columns; col++)
    {
        for (int row = 0; row <= rows - verticalWin; row++)
        {
            bool win = true;
            for (int k = 0; k < verticalWin; k++)
            {
                if (board[(row + k) * columns + col] != player)
                {
                    win = false;
                    break;
                }
            }
            if (win)
                return true;
        }
    }

    // Diagonal check (top-left to bottom-right)
    for (int row = 0; row <= rows - horizontalWin; row++)
    {
        for (int col = 0; col <= columns - horizontalWin; col++)
        {
            bool win = true;
            for (int k = 0; k < horizontalWin; k++)
            {
                if (board[(row + k) * columns + col + k] != player)
                {
                    win = false;
                    break;
                }
            }
            if (win)
                return true;
        }
    }

    // Diagonal check (top-right to bottom-left)
    for (int row = 0; row <= rows - horizontalWin; row++)
    {
        for (int col = horizontalWin - 1; col < columns; col++)
        {
            bool win = true;
            for (int k = 0; k < horizontalWin; k++)
            {
                if (board[(row + k) * columns + col - k] != player)
                {
                    win = false;
                    break;
                }
            }
            if (win)
                return true;
        }
    }

    return false;
}

int main(int argc, char *argv[])
{
    int rows = DEFAULT_ROWS;
    int columns = DEFAULT_COLUMNS;

    if (argc >= 2)
    {
        rows = atoi(argv[1]);
        columns = atoi(argv[1]);
    }

    printf("Using a %dx%d board.\n", rows, columns);

    // Determine available CUDA devices
    int nDevices;
    cudaGetDeviceCount(&nDevices);

    // Assign devices for Player 1 and Player 2
    int device_1 = 0;
    int device_2 = 0;

    if (nDevices >= 2)
    {
        device_2 = 1;
    }
    else
    {
        printf("Warning: Fewer than 2 GPUs available. Both players will use GPU 0.\n");
    }

    // Allocate and initialize boards on GPUs
    int *board_gpu1, *board_gpu2;
    cudaSetDevice(device_1);
    cudaMalloc(&board_gpu1, rows * columns * sizeof(int));
    cudaMemset(board_gpu1, 0, rows * columns * sizeof(int));

    cudaSetDevice(device_2);
    cudaMalloc(&board_gpu2, rows * columns * sizeof(int));
    cudaMemset(board_gpu2, 0, rows * columns * sizeof(int));

    int *host_board = (int *)malloc(rows * columns * sizeof(int)); // Host memory for copying the board
    cudaError_t err;

    int maxRounds = rows * columns;
    int round = 0;
    int currentPlayer = 1;

    while (true)
    {
        if (currentPlayer == 1)
        {
            unsigned long long seed = time(NULL) + round;
            randomMove<<<1, 1>>>(board_gpu1, currentPlayer, rows, columns, seed);
            err = cudaDeviceSynchronize();
            if (err != cudaSuccess)
            {
                printf("CUDA error after randomMove: %s\n", cudaGetErrorString(err));
                return -1;
            }

            // Copy board from GPU1 to host and check for win
            cudaMemcpy(host_board, board_gpu1, rows * columns * sizeof(int), cudaMemcpyDeviceToHost);
            printBoard(host_board, rows, columns);
            if (checkWin(host_board, currentPlayer, rows, columns))
            {
                printf("Player %d wins!\n", currentPlayer);
                break;
            }

            // Copy board from GPU1 to GPU2
            cudaMemcpy(board_gpu2, board_gpu1, rows * columns * sizeof(int), cudaMemcpyDeviceToDevice);
            currentPlayer = 2;
        }
        else
        {
            lookaheadMove<<<1, 1>>>(board_gpu2, currentPlayer, rows, columns);
            err = cudaDeviceSynchronize();
            if (err != cudaSuccess)
            {
                printf("CUDA error after lookaheadMove: %s\n", cudaGetErrorString(err));
                break;
            }

            // Copy board from GPU2 to host and check for win
            cudaMemcpy(host_board, board_gpu2, rows * columns * sizeof(int), cudaMemcpyDeviceToHost);
            printBoard(host_board, rows, columns);
            if (checkWin(host_board, currentPlayer, rows, columns))
            {
                printf("Player %d wins!\n", currentPlayer);
                break;
            }

            // Copy board from GPU2 to GPU1
            cudaMemcpy(board_gpu1, board_gpu2, rows * columns * sizeof(int), cudaMemcpyDeviceToDevice);
            currentPlayer = 1;
        }

        round++;
        if (round >= maxRounds)
        {
            printf("Maximum rounds reached. The game is a draw!\n");
            break;
        }
    }

    // Free GPU and host memory
    cudaFree(board_gpu1);
    cudaFree(board_gpu2);
    free(host_board);

    printf("Game completed successfully.\n");
    return 0;
}
