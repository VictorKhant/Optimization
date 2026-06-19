SHELL := /bin/bash

CC = gcc
MPICC = mpicc
NPROC = 4
CFLAGS = -std=c99 -g -c -Wall -O1
CFLAGS_OPTIMIZE = -fopenmp -mavx -mavx2 -mfma
LDFLAGS = -fopenmp

# When building the Open MPI coordinator, compile and link with mpicc and run
# the binary under mpirun. Everything else uses the regular compiler.
ifeq ($(COORDINATOR),mpi)
COORD_CC = $(MPICC)
COMPUTE_CC = $(MPICC)
LINKER = $(MPICC)
RUNNER = mpirun -np $(NPROC)
else
COORD_CC = $(CC)
COMPUTE_CC = $(CC)
LINKER = $(CC)
RUNNER =
endif

# Default
default: | ensure_compute ensure_coordinator
default: convolve_$(COORDINATOR)_$(COMPUTE)

# Utils
.PHONY: check-hashes ensure_compute ensure_coordinator ensure_input
check-hashes:
	@bash tools/run_python.sh tools/check_hashes.py

ensure_compute:
ifeq ($(COMPUTE),)
	@echo "COMPUTE must be set"
	@exit 1
endif

ensure_coordinator:
ifeq ($(COORDINATOR),)
	@echo "COORDINATOR must be set"
	@exit 1
endif

ensure_test:
ifeq ($(TEST),)
	@echo "TEST must be set"
	@exit 1
endif

# Task specific targets
.PHONY: task_1 task_2 task_3

task_1: | ensure_test
	make run COORDINATOR=naive COMPUTE=naive TEST=$(TEST)

task_2: | ensure_test
	make run COORDINATOR=naive COMPUTE=optimized TEST=$(TEST)

task_3: | ensure_test
	make run COORDINATOR=mpi COMPUTE=optimized_mpi TEST=$(TEST)

# C targets
src/coordinator_$(COORDINATOR).o: src/coordinator_$(COORDINATOR).c
	$(COORD_CC) $(CFLAGS) -c $< -o $@

src/compute_$(COMPUTE).o: src/compute_$(COMPUTE).c
	$(COMPUTE_CC) $(CFLAGS) $(CFLAGS_OPTIMIZE) -c $< -o $@

convolve_$(COORDINATOR)_$(COMPUTE): | check-hashes
convolve_$(COORDINATOR)_$(COMPUTE): src/compute_$(COMPUTE).o src/coordinator_$(COORDINATOR).o src/io.o
	$(LINKER) $(LDFLAGS) $^ -o $@

run: | ensure_compute ensure_coordinator ensure_test
run: convolve_$(COORDINATOR)_$(COMPUTE)
	@bash tools/run_python.sh tools/create_tests.py $(shell basename $(TEST))
	@rm -f $(TEST)/*/out.bin
	@bash tools/run_test.sh $(RUNNER) ./convolve_$(COORDINATOR)_$(COMPUTE) $(TEST)/input.txt
	@bash tools/check_output.sh $(TEST)
	@if [ -f $(TEST)/gifs.json ]; then bash tools/run_python.sh tools/results_to_gif.py $(TEST); fi

# Clean
clean:
	rm -f src/coordinator_*.o src/compute_*.o
	rm -f convolve_*
