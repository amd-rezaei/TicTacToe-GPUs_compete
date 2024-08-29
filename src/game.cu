#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <time.h>

#define ROWS 3
#define COLUMNS 3

// Kernel for Player 1: Random Strategy
__global__ void randomMove(int *board, int player, unsigned long long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0)
    {
        curandState_t state;
        curand_init(seed, idx, 0, &state);

        int col = curand(&state) % COLUMNS; // Random column
        while (true)
        {
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
            // Move to the next column
            col = (col + 1) % COLUMNS;
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


// Check for a win condition, returns true if player wins
bool checkWin(int *board, int player)
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

// Display the board state
void printBoard(int *board)
{
    for (int i = 0; i < ROWS; i++)
    {
        for (int j = 0; j < COLUMNS; j++)
        {
            int token = board[i * COLUMNS + j];
            printf("%c ", (token == 0) ? '.' : (token == 1 ? 'X' : 'O'));
        }
        printf("\n");
    }
    printf("\n");
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

    int maxRounds = ROWS * COLUMNS; // Maximum number of moves (3 rows * 3 columns)
    int round = 0;
    int currentPlayer = 1;

    while (true)
    {
        if (currentPlayer == 1)
        {
            randomMove<<<1, 1>>>(board, currentPlayer, time(NULL)); // Single thread
            cudaDeviceSynchronize();
            printBoard(board);
            if (checkWin(board, currentPlayer))
            {
                printf("Player %d wins!\n", currentPlayer);
                break;
            }
            currentPlayer = 2;
        }
        else
        {
            lookaheadMove<<<1, 1>>>(board, currentPlayer); // Single thread
            cudaDeviceSynchronize();
            printBoard(board);
            if (checkWin(board, currentPlayer))
            {
                printf("Player %d wins!\n", currentPlayer);
                break;
            }
            currentPlayer = 1;
        }

        round++;
        if (round >= maxRounds)
        {
            printf("Maximum rounds reached. The game is a draw!\n");
            break;
        }
    }

    cudaFree(board);
    return 0;
}
