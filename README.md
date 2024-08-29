
# CUDA-based Tic-Tac-Toe Game

This project implements a CUDA-based version of the classic Tic-Tac-Toe game, where two AI players compete against each other using distinct strategies. The game leverages NVIDIA GPUs to handle game state management and AI decision-making, showcasing the use of GPU computation to accelerate these processes.

## Project Overview

The Tic-Tac-Toe game is implemented using CUDA, with the ability to dynamically select between available GPUs. If two or more GPUs are detected, the game assigns one GPU to each player; if only one GPU is available, both players will use the same GPU. The game features two AI strategies:
- **Random Strategy (Player 1)**: Player 1 randomly selects a column to place its token, ensuring a non-deterministic game play.
- **Lookahead Strategy (Player 2)**: Player 2 implements a basic lookahead mechanism, attempting to block the opponent or create a winning sequence.

## Getting Started

### Prerequisites

To run this project, ensure that your environment meets the following requirements:
- **CUDA Toolkit**: Ensure that the CUDA Toolkit is installed on your system. This project was tested with CUDA 11.4, but newer versions should also work.
- **NVIDIA GPU**: The game requires an NVIDIA GPU with the appropriate CUDA-capable drivers installed.

### Compilation and Execution

1. **Clone the Repository**: First, clone the repository to your local machine:
    ```bash
    git clone https://github.com/your-repo/cuda-tic-tac-toe.git
    cd cuda-tic-tac-toe/src
    ```

2. **Compile the Code**: Navigate to the `src/` directory and compile the code using the `nvcc` compiler:
    ```bash
    nvcc -o tic_tac_toe game.cu
    ```

3. **Run the Game**: After compilation, you can run the game using:
    ```bash
    ./tic_tac_toe
    ```

### How It Works

- **Dynamic GPU Selection**: The program automatically detects the number of available GPUs. If there are at least two GPUs, each player is assigned to a different GPU. If only one GPU is available, both players share the same GPU, ensuring compatibility with various hardware configurations.
- **Game Logic**: The game alternates between the two AI strategies:
  - **Player 1 (Random Strategy)**: Randomly selects a column to place its token.
  - **Player 2 (Lookahead Strategy)**: Attempts to block Player 1 or place a winning token by looking ahead one move.

### Example Output

When running the game, you might see output similar to this:

```
Player 1 move completed
Board copied from GPU1 to host
Current Board State:
0 0 0 
1 0 0 
0 0 0 

Player 2 move completed
Board copied from GPU2 to host
Current Board State:
0 0 0 
1 0 0 
0 0 2 

Player 1 move completed
...
```

### Additional Notes

- **Error Handling**: The program includes basic error handling to ensure smooth operation even in the case of unexpected issues, such as insufficient GPUs or memory allocation errors.
- **Scalability**: Although this project implements a simple 3x3 Tic-Tac-Toe game, the logic can be extended to support more complex games, like Connect 4, with minimal modifications.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
