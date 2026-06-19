#include <mpi.h>

#include "coordinator.h"

int main(int argc, char *argv[]) {
  if (argc < 2) {
    printf("Error: not enough arguments\n");
    printf("Usage: %s [path_to_task_list]\n", argv[0]);
    return -1;
  }

  // Read and parse task list file (every process reads it independently so
  // each one knows the full set of tasks without any communication).
  int num_tasks;
  task_t **tasks;
  if (read_tasks(argv[1], &num_tasks, &tasks)) {
    printf("Error reading task list from %s\n", argv[1]);
    return -1;
  }

  // Initialize MPI and find out which process we are (procID) and how many
  // processes there are in total (num_procs).
  MPI_Init(&argc, &argv);
  int procID, num_procs;
  MPI_Comm_rank(MPI_COMM_WORLD, &procID);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

  // Statically partition the tasks across processes in a round-robin fashion.
  // Process i handles tasks i, i + num_procs, i + 2 * num_procs, ...
  // This needs no inter-process communication and spreads the work evenly.
  int err = 0;
  for (int i = procID; i < num_tasks; i += num_procs) {
    if (execute_task(tasks[i])) {
      printf("Task %d failed\n", i);
      err = -1;
      break;
    }
  }

  // Clean up the task list.
  for (int i = 0; i < num_tasks; i++) {
    free(tasks[i]->path);
    free(tasks[i]);
  }
  free(tasks);

  MPI_Finalize();
  return err;
}
