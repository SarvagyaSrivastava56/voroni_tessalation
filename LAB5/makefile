CC = mpicc
CFLAGS = -Wall -O2
TARGET = perfect_search
SRCS = main.c perfect_worker.c
NP = 2

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRCS)

run: $(TARGET)
	@mpirun --oversubscribe -np $(NP) ./$(TARGET)

clean:
	rm -f $(TARGET)