CC = gcc
CFLAGS = -std=c99 -Wall -Wextra -Wpedantic -g -fopenmp -D_POSIX_C_SOURCE=199309L
LDFLAGS = -lm -lpthread -lgomp

# Source directories
MATRIX_DIR = matrix
NEURAL_DIR = neural
SOCKET_DIR = socket
UTIL_DIR = util
APPS_DIR = apps

# Source files
MATRIX_SRCS = $(wildcard $(MATRIX_DIR)/*.c)
NEURAL_SRCS = $(wildcard $(NEURAL_DIR)/*.c)
SOCKET_SRCS = $(wildcard $(SOCKET_DIR)/*.c)
UTIL_SRCS = $(wildcard $(UTIL_DIR)/*.c)

# Object files
MATRIX_OBJS = $(MATRIX_SRCS:.c=.o)
NEURAL_OBJS = $(NEURAL_SRCS:.c=.o)
SOCKET_OBJS = $(SOCKET_SRCS:.c=.o)
UTIL_OBJS = $(UTIL_SRCS:.c=.o)

# Common objects
COMMON_OBJS = $(MATRIX_OBJS) $(NEURAL_OBJS) $(SOCKET_OBJS) $(UTIL_OBJS)

# Targets
all: setup parameter_server worker app

# Parameter server
parameter_server: $(COMMON_OBJS) $(APPS_DIR)/parameter_server.o
	$(CC) $^ -o $@ $(LDFLAGS)

# Worker
worker: $(COMMON_OBJS) $(APPS_DIR)/worker.o  
	$(CC) $^ -o $@ $(LDFLAGS)

# Legacy app (for backward compatibility)
app: $(COMMON_OBJS) train.o
	$(CC) $^ -o $@ $(LDFLAGS)

# Server target alias
server: parameter_server

# Object file compilation
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean
clean:
	rm -f $(COMMON_OBJS) $(APPS_DIR)/*.o train.o
	rm -f parameter_server worker app

# Create results directories
setup:
	mkdir -p results/server_logs results/worker1_results results/worker2_results

.PHONY: all server worker clean setup