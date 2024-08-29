#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <time.h>

#define ROWS 3
#define COLUMNS 3

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

__global__ void lookaheadMove(int *board, int player)
{
    int col = threadIdx.x % COLUMNS;

    // Simulate the opponent's move (Player 1)
    int opponent = 3 - player;

    // Check if the opponent can win by placing a piece in the current column
    for (int i = ROWS - 1; i >= 0; i--)
    {
        if (board[col * ROWS + i] == 0)
        {
            // Temporarily place the opponent's piece in this column
            board[col * ROWS + i] = opponent;

            // Inline win check
            bool win = false;

            // Horizontal check
            for (int row = 0; row < ROWS; row++)
            {
                int count = 0;
                for (int j = 0; j < COLUMNS; j++)
                {
                    if (board[j * ROWS + row] == opponent)
                        count++;
                    else
                        count = 0;
                    if (count == 4)
                    {
                        win = true;
                        break;
                    }
                }
                if (win)
                    break;
            }

            // Vertical check
            for (int j = 0; j < COLUMNS; j++)
            {
                int count = 0;
                for (int row = 0; row < ROWS; row++)
                {
                    if (board[j * ROWS + row] == opponent)
                        count++;
                    else
                        count = 0;
                    if (count == 4)
                    {
                        win = true;
                        break;
                    }
                }
                if (win)
                    break;
            }

            // Diagonal checks (down-right and down-left)
            for (int row = 0; row < ROWS - 3; row++)
            {
                for (int j = 0; j < COLUMNS - 3; j++)
                {
                    if (board[j * ROWS + row] == opponent &&
                        board[(j + 1) * ROWS + (row + 1)] == opponent &&
                        board[(j + 2) * ROWS + (row + 2)] == opponent &&
                        board[(j + 3) * ROWS + (row + 3)] == opponent)
                    {
                        win = true;
                        break;
                    }
                }
                if (win)
                    break;
            }

            for (int row = 0; row < ROWS - 3; row++)
            {
                for (int j = 3; j < COLUMNS; j++)
                {
                    if (board[j * ROWS + row] == opponent &&
                        board[(j - 1) * ROWS + (row + 1)] == opponent &&
                        board[(j - 2) * ROWS + (row + 2)] == opponent &&
                        board[(j - 3) * ROWS + (row + 3)] == opponent)
                    {
                        win = true;
                        break;
                    }
                }
                if (win)
                    break;
            }

            // If the opponent would win, place the current player's piece here to block
            if (win)
            {
                board[col * ROWS + i] = player;
                return;
            }

            // Revert the temporary move
            board[col * ROWS + i] = 0;
            break; // Stop checking further rows in this column
        }
    }

    // If no winning move to block, make a random move (fallback)
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

    int maxRounds = ROWS * COLUMNS; // Maximum number of moves (6 rows * 7 columns)
    int round = 0;
    int currentPlayer = 1;
    while (true)
    {
        if (currentPlayer == 1)
        {
            
            randomMove<<<1, COLUMNS>>>(board, currentPlayer);
            currentPlayer = 2;
        }
        else
        {
            
            lookaheadMove<<<1, COLUMNS>>>(board, currentPlayer);
            currentPlayer = 1;
        }
        cudaDeviceSynchronize();
        printBoard(board);

        if (checkWin(board, currentPlayer))
        {
            printf("Player %d wins!\n", currentPlayer);
            break;
        }

        currentPlayer = 3 - currentPlayer; // Toggle between player 1 and 2

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
