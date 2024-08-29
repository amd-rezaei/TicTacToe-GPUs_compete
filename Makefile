# Makefile for compiling CUDA code

# Compiler
NVCC = nvcc

# Target executable
TARGET = game

# Source files
SRCS = src/game.cu

# Compiler flags
CFLAGS = -O2 -arch=sm_30

# Linker flags
LDFLAGS = 

# Default rule to compile and link the program
all: $(TARGET)

$(TARGET): $(SRCS)
	$(NVCC) $(CFLAGS) $(SRCS) -o $(TARGET) $(LDFLAGS)

# Clean rule to remove generated files
clean:
	rm -f $(TARGET)
