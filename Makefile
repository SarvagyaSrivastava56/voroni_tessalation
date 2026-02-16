# Makefile for correlation program

# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++11 -Wall -Wextra -O2 -fopenmp
LDFLAGS = -lpsapi
TARGET = correlate_program

# Source files
SOURCES = main.cpp correlate.cpp
OBJECTS = $(SOURCES:.cpp=.o)

# Default target
all: $(TARGET)

# Link the executable
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Compile source files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean build artifacts
clean:
	del /Q *.o $(TARGET).exe 2>nul || true

# Run the program
run: $(TARGET)
	./$(TARGET).exe

# Debug build
debug: CXXFLAGS += -g -DDEBUG
debug: $(TARGET)

.PHONY: all clean run debug
