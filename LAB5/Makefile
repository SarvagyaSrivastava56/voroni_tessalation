CC = mpicc
CFLAGS = -Wall -O2
TARGET = daxpy_mpi
SRC = daxpy_mpi.c
NP = 8

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC)

run: $(TARGET)
	@mpirun -np $(NP) -oversubscribe ./$(TARGET)

clean:
	rm -f $(TARGET)