# Compiler and flags
CC = mpicc
CFLAGS = -fopenmp
LDFLAGS = -lm

# Sources and headers
C_SOURCES = $(wildcard matrix/*.c neural/*.c util/*.c)
HEADERS = $(wildcard matrix/*.h neural/*.h util/*.h *.h)

# Object files
OBJ = $(C_SOURCES:.c=.o)

# Executables
TRAIN = train
PREDICT = predict

# Default target
all: $(TRAIN) $(PREDICT)

# Build train
$(TRAIN): train.c $(OBJ)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

# Build predict
$(PREDICT): predict.c $(OBJ)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

# Object file rule
%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

# Run with mpirun
run_train: $(TRAIN)
	mpirun -np 4 ./$(TRAIN)

run_predict: $(PREDICT)
	mpirun -np 4 ./$(PREDICT)

# Clean target
clean:
	rm -f matrix/*.o neural/*.o util/*.o *.o $(TRAIN) $(PREDICT)
