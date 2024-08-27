# CUDA-based Connect 4 Game

This project is a CUDA-based implementation of the classic Connect 4 game, where two AI players compete against each other using different strategies. The game is designed to run on NVIDIA GPUs using the CUDA programming model. It demonstrates the use of basic AI strategies and GPU computation to manage game state and determine player moves.

## Project Description

The Connect 4 game is implemented in CUDA and capable of selecting between available GPUs dynamically. If more than one GPU is available, one is chosen at random; otherwise, the first GPU is used. The game supports two types of strategies:
- **Random Strategy (Player 1)**: Chooses a column at random to place the token.
- **Lookahead Strategy (Player 2)**: Implements a basic lookahead to block the opponent or connect four tokens.

## Getting Started

### Prerequisites
Ensure you have the following installed:
- CUDA Toolkit (Tested with CUDA 11.4)
- NVIDIA GPU with proper driver installed

### Compilation and Execution
Navigate to the `src/` directory and compile the code using the following command:

```bash
nvcc -o connect4 main.cu
