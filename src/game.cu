#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <time.h>

#define ROWS 3
#define COLUMNS 3

// Kernel for Player 1: Random Move Strategy
__global__ void randomMove(int *board, int player, unsigned long long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0)
    {
        curandState_t state;
        curand_init(seed, idx, 0, &state);

        int col = curand(&state) % COLUMNS;
        int startCol = col;

        while (true)
        {
            if (board[col * ROWS] == 0)
            {
                for (int i = ROWS - 1; i >= 0; i--)
                {
                    if (board[col * ROWS + i] == 0)
                    {
                        board[col * ROWS + i] = player;
                        return;
                    }
                }
            }
            col = (col + 1) % COLUMNS;
            if (col == startCol)
                break; // Break if we've checked all columns
        }
    }
}

// Kernel for Player 2: Lookahead Strategy
__global__ void lookaheadMove(int *board, int player)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0)
    {
        int opponent = 3 - player;

        for (int col = 0; col < COLUMNS; col++)
        {
            for (int i = ROWS - 1; i >= 0; i--)
            {
                if (board[col * ROWS + i] == 0)
                {
                    board[col * ROWS + i] = opponent;
                    bool win = false;

                    // Horizontal check
                    for (int row = 0; row < ROWS; row++)
                    {
                        if (board[row * COLUMNS] == opponent &&
                            board[row * COLUMNS + 1] == opponent &&
                            board[row * COLUMNS + 2] == opponent)
                        {
                            win = true;
                        }
                    }

                    // Vertical check
                    for (int colcnt = 0; colcnt < COLUMNS; colcnt++)
                    {
                        if (board[colcnt] == opponent &&
                            board[colcnt + COLUMNS] == opponent &&
                            board[colcnt + 2 * COLUMNS] == opponent)
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
                        board[col * ROWS + i] = player;
                        return;
                    }

                    board[col * ROWS + i] = 0;
                    break;
                }
            }
        }

        // If no winning move to block, make a random move
        for (int col = 0; col < COLUMNS; col++)
        {
            if (board[col * ROWS] == 0)
            {
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
    }
}

// Utility function to print the current board state
void printBoard(const int *board)
{
    printf("Current Board State:\n");
    for (int i = 0; i < ROWS; ++i)
    {
        for (int j = 0; j < COLUMNS; ++j)
        {
            printf("%d ", board[i * COLUMNS + j]);
        }
        printf("\n");
    }
    printf("\n");
}

// Utility function to check if a player has won
bool checkWin(const int *board, int player)
{
    // Horizontal check
    for (int row = 0; row < ROWS; row++)
    {
        if (board[row * COLUMNS] == player &&
            board[row * COLUMNS + 1] == player &&
            board[row * COLUMNS + 2] == player)
        {
            return true;
        }
    }

    // Vertical check
    for (int col = 0; col < COLUMNS; col++)
    {
        if (board[col] == player &&
            board[col + COLUMNS] == player &&
            board[col + 2 * COLUMNS] == player)
        {
            return true;
        }
    }

    // Diagonal check (top-left to bottom-right)
    if (board[0] == player &&
        board[4] == player &&
        board[8] == player)
    {
        return true;
    }

    // Diagonal check (top-right to bottom-left)
    if (board[2] == player &&
        board[4] == player &&
        board[6] == player)
    {
        return true;
    }

    return false;
}

int main()
{
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
    cudaMalloc(&board_gpu1, ROWS * COLUMNS * sizeof(int));
    cudaMemset(board_gpu1, 0, ROWS * COLUMNS * sizeof(int));

    cudaSetDevice(device_2);
    cudaMalloc(&board_gpu2, ROWS * COLUMNS * sizeof(int));
    cudaMemset(board_gpu2, 0, ROWS * COLUMNS * sizeof(int));

    int host_board[ROWS * COLUMNS]; // Host memory for copying the board
    cudaError_t err;

    int maxRounds = ROWS * COLUMNS;
    int round = 0;
    int currentPlayer = 1;

    while (true)
    {
        if (currentPlayer == 1)
        {
            unsigned long long seed = time(NULL) + round;
            randomMove<<<1, 1>>>(board_gpu1, currentPlayer, seed);
            err = cudaDeviceSynchronize();
            if (err != cudaSuccess)
            {
                printf("CUDA error after randomMove: %s\n", cudaGetErrorString(err));
                return -1;
            }

            // Copy board from GPU1 to host and check for win
            cudaMemcpy(host_board, board_gpu1, ROWS * COLUMNS * sizeof(int), cudaMemcpyDeviceToHost);
            printBoard(host_board);
            if (checkWin(host_board, currentPlayer))
            {
                printf("Player %d wins!\n", currentPlayer);
                break;
            }

            // Copy board from GPU1 to GPU2
            cudaMemcpy(board_gpu2, board_gpu1, ROWS * COLUMNS * sizeof(int), cudaMemcpyDeviceToDevice);
            currentPlayer = 2;
        }
        else
        {
            lookaheadMove<<<1, 1>>>(board_gpu2, currentPlayer);
            err = cudaDeviceSynchronize();
            if (err != cudaSuccess)
            {
                printf("CUDA error after lookaheadMove: %s\n", cudaGetErrorString(err));
                break;
            }

            // Copy board from GPU2 to host and check for win
            cudaMemcpy(host_board, board_gpu2, ROWS * COLUMNS * sizeof(int), cudaMemcpyDeviceToHost);
            printBoard(host_board);
            if (checkWin(host_board, currentPlayer))
            {
                printf("Player %d wins!\n", currentPlayer);
                break;
            }

            // Copy board from GPU2 to GPU1
            cudaMemcpy(board_gpu1, board_gpu2, ROWS * COLUMNS * sizeof(int), cudaMemcpyDeviceToDevice);
            currentPlayer = 1;
        }

        round++;
        if (round >= maxRounds)
        {
            printf("Maximum rounds reached. The game is a draw!\n");
            break;
        }
    }

    // Free GPU memory
    cudaFree(board_gpu1);
    cudaFree(board_gpu2);

    printf("Game completed successfully.\n");
    return 0;
}
