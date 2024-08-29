# Makefile for compiling CUDA code

# Compiler
NVCC = nvcc

# Target executable directory
SRC_DIR = src

# Target executable
TARGET = $(SRC_DIR)/game

# Source files
SRCS = $(SRC_DIR)/game.cu

# Compiler flags
# Adding -g for debugging symbols and -O0 to disable optimizations
CFLAGS = -g -O0 -arch=sm_89

# Linker flags
LDFLAGS = 

# Default rule to compile and link the program
all: $(TARGET)

$(TARGET): $(SRCS)
	$(NVCC) $(CFLAGS) $(SRCS) -o $(TARGET) $(LDFLAGS)

# Clean rule to remove generated files
clean:
	rm -f $(TARGET)
