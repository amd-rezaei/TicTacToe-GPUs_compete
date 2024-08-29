
# CUDA-based Tic-Tac-Toe Game

This project implements a CUDA-based Tic-Tac-Toe game (or any Connect N variant), where two AI players compete against each other using different strategies. The game is designed to run on NVIDIA GPUs using the CUDA programming model. This project demonstrates the use of basic AI strategies and GPU computation to manage game state and determine player moves.

## Project Description

The game supports two AI strategies:
- **Random Strategy (Player 1)**: Chooses a column randomly to place the token.
- **Lookahead Strategy (Player 2)**: Implements a basic lookahead to block the opponent or connect a specified number of tokens (based on the board's size).

### Key Features
- **Dynamic GPU Selection**: The game checks the available CUDA-enabled GPUs. If two or more GPUs are available, it assigns one to each player. If fewer than two GPUs are available, both players use the first GPU.
- **Flexible Board Size**: The board size can be specified via a command-line argument. The game supports square board sizes (e.g., 3x3, 4x4). The win condition dynamically adapts to the size of the board.

## Getting Started

### Prerequisites

Ensure you have the following installed:
- CUDA Toolkit (Tested with CUDA 11.4 or later)
- NVIDIA GPU with the proper driver installed

### Compilation

Navigate to the project directory and compile the code using the following command:

```bash
nvcc -o tic_tac_toe game.cu
```

### Execution

To run the game with a default 3x3 board:

```bash
./tic_tac_toe
```

To run with a custom square board size, such as 4x4:

```bash
./tic_tac_toe 4
```

### Example Output

Here’s an example of how the output might look for a 4x4 board:

```plaintext
Using a 4x4 board.
Player 1 move completed
Board copied from GPU1 to host
Current Board State:
0 0 0 0
0 0 0 0
0 0 0 0
1 0 0 0

Player changed to 2
Player 2 move completed
Current Board State:
0 0 0 0
0 0 0 0
0 0 0 0
1 2 0 0

Player changed to 1
...
Player 1 wins!
Game completed successfully.
```

### Customization

The game logic allows you to easily adjust the board size, making the game more versatile for different configurations. Simply pass the desired size as a command-line argument.

### Known Issues

- Ensure the board size is appropriate for the number of moves you want to play. The win condition is tied to the number of rows and columns, so a board like 4x4 requires four tokens in a row, column, or diagonal to win.

### Debugging

The game includes detailed debugging output that shows when and where a player wins. This output is helpful for understanding the internal game logic and ensuring correct behavior.

## Authors

This project was developed as part of a CUDA programming exercise. The primary focus was on leveraging GPU parallelism for simple AI-driven gameplay.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
