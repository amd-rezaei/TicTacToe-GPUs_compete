#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <time.h>

#define DEFAULT_ROWS 3
#define DEFAULT_COLUMNS 3

// Kernel for Player 1: Makes a random move
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
                break; // All columns checked
        }
    }
}

// Kernel for Player 2: Tries to block the opponent's winning move
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

                    // Check if opponent would win
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

                    // Diagonal checks
                    if (board[0] == opponent && board[4] == opponent && board[8] == opponent)
                    {
                        win = true;
                    }
                    if (board[2] == opponent && board[4] == opponent && board[6] == opponent)
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

        // No block needed, make a move
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

void printBoard(const int *board, int rows, int columns)
{
    printf("Board:\n");
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < columns; ++j)
        {
            if (board[i * columns + j] == 1){
                printf("%s ", "O");
            }
            else if (board[i * columns + j] == 2){
                printf("%s ", "X");
            }
            else{
                printf("%s ", "-");
            }
        }
        printf("\n");
    }
    printf("\n");
}

bool checkWin(const int *board, int player, int rows, int columns)
{
    int winCondition = rows;

    for (int row = 0; row < rows; row++)
    {
        for (int col = 0; col <= columns - winCondition; col++)
        {
            bool win = true;
            for (int k = 0; k < winCondition; k++)
            {
                if (board[row * columns + col + k] != player)
                {
                    win = false;
                    break;
                }
            }
            if (win)
            {
                printf("Player %d wins horizontally at row %d, column %d\n", player, row, col);
                return true;
            }
        }
    }

    for (int col = 0; col < columns; col++)
    {
        for (int row = 0; row <= rows - winCondition; row++)
        {
            bool win = true;
            for (int k = 0; k < winCondition; k++)
            {
                if (board[(row + k) * columns + col] != player)
                {
                    win = false;
                    break;
                }
            }
            if (win)
            {
                printf("Player %d wins vertically at row %d, column %d\n", player, row, col);
                return true;
            }
        }
    }

    for (int row = 0; row <= rows - winCondition; row++)
    {
        for (int col = 0; col <= columns - winCondition; col++)
        {
            bool win = true;
            for (int k = 0; k < winCondition; k++)
            {
                if (board[(row + k) * columns + col + k] != player)
                {
                    win = false;
                    break;
                }
            }
            if (win)
            {
                printf("Player %d wins diagonally (\\) at row %d, column %d\n", player, row, col);
                return true;
            }
        }
    }

    for (int row = 0; row <= rows - winCondition; row++)
    {
        for (int col = winCondition - 1; col < columns; col++)
        {
            bool win = true;
            for (int k = 0; k < winCondition; k++)
            {
                if (board[(row + k) * columns + col - k] != player)
                {
                    win = false;
                    break;
                }
            }
            if (win)
            {
                printf("Player %d wins diagonally (/) at row %d, column %d\n", player, row, col);
                return true;
            }
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

    printf("Board size: %dx%d\n", rows, columns);

    int nDevices;
    cudaGetDeviceCount(&nDevices);

    int device_1 = 0;
    int device_2 = (nDevices >= 2) ? 1 : 0;

    if (nDevices < 2)
    {
        printf("Warning: Only one GPU available.\n");
    }

    int *board_gpu1, *board_gpu2;
    cudaSetDevice(device_1);
    cudaMalloc(&board_gpu1, rows * columns * sizeof(int));
    cudaMemset(board_gpu1, 0, rows * columns * sizeof(int));

    cudaSetDevice(device_2);
    cudaMalloc(&board_gpu2, rows * columns * sizeof(int));
    cudaMemset(board_gpu2, 0, rows * columns * sizeof(int));

    int *host_board = (int *)malloc(rows * columns * sizeof(int));
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
                printf("CUDA error in randomMove: %s\n", cudaGetErrorString(err));
                return -1;
            }

            cudaMemcpy(host_board, board_gpu1, rows * columns * sizeof(int), cudaMemcpyDeviceToHost);
            printBoard(host_board, rows, columns);
            if (checkWin(host_board, currentPlayer, rows, columns))
            {
                printf("Player %d wins!\n", currentPlayer);
                break;
            }

            cudaMemcpy(board_gpu2, board_gpu1, rows * columns * sizeof(int), cudaMemcpyDeviceToDevice);
            currentPlayer = 2;
        }
        else
        {
            lookaheadMove<<<1, 1>>>(board_gpu2, currentPlayer, rows, columns);
            err = cudaDeviceSynchronize();
            if (err != cudaSuccess)
            {
                printf("CUDA error in lookaheadMove: %s\n", cudaGetErrorString(err));
                break;
            }

            cudaMemcpy(host_board, board_gpu2, rows * columns * sizeof(int), cudaMemcpyDeviceToHost);
            printBoard(host_board, rows, columns);
            if (checkWin(host_board, currentPlayer, rows, columns))
            {
                printf("Player %d wins!\n", currentPlayer);
                break;
            }

            cudaMemcpy(board_gpu1, board_gpu2, rows * columns * sizeof(int), cudaMemcpyDeviceToDevice);
            currentPlayer = 1;
        }

        round++;
        if (round >= maxRounds)
        {
            printf("Draw! No moves left.\n");
            break;
        }
    }

    cudaFree(board_gpu1);
    cudaFree(board_gpu2);
    free(host_board);

    printf("Game over.\n");
    return 0;
}
