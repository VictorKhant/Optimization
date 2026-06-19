#include "compute.h"

// Computes the convolution of two matrices
int convolve(matrix_t *a_matrix, matrix_t *b_matrix, matrix_t **output_matrix) {
  // TODO: convolve matrix a and matrix b, and store the resulting matrix in
  // output_matrix
    if(a_matrix->rows < b_matrix->rows || a_matrix->cols < b_matrix->cols){
        return -1;
    }
    int temp = 0;
    int size = b_matrix->rows * b_matrix->cols; 
    for(uint32_t i = 0; i < size / 2; ++i) {
        temp = b_matrix->data[i];
        b_matrix->data[i] = b_matrix->data[size - i - 1];
        b_matrix->data[size - i - 1] = temp;
    }
    uint32_t newRows = a_matrix->rows - b_matrix->rows + 1;
    uint32_t newCols = a_matrix->cols - b_matrix->cols + 1;
    
    if(newRows <= 0 || newCols <= 0){
        return -1;
    }
    *output_matrix = (matrix_t *) malloc(sizeof(matrix_t));
    (*output_matrix)->rows = newRows;
    (*output_matrix)->cols = newCols;
    (*output_matrix)->data = (int32_t*) malloc(newRows * newCols * sizeof(int32_t));

    if(!(*output_matrix)->data){
        return -1;
    }

    for(uint32_t i = 0; i < newRows; i++){
        for(uint32_t j = 0; j < newCols; j++){
            int sum = 0;
            for(uint32_t m = 0; m < b_matrix->rows; m++){
                for(uint32_t n = 0; n < b_matrix->cols; n++){
                    uint32_t a_index = (i + m) * a_matrix->cols + (j + n);
                    uint32_t b_index = m* b_matrix->cols + n;
                    sum += a_matrix->data[a_index] * b_matrix->data[b_index];
                }
            }
            (*output_matrix)->data[i * newCols + j] = sum;
        }
    }
  return 0;
}

// Executes a task
int execute_task(task_t *task) {
  matrix_t *a_matrix, *b_matrix, *output_matrix;

  char *a_matrix_path = get_a_matrix_path(task);
  if (read_matrix(a_matrix_path, &a_matrix)) {
    printf("Error reading matrix from %s\n", a_matrix_path);
    return -1;
  }
  free(a_matrix_path);

  char *b_matrix_path = get_b_matrix_path(task);
  if (read_matrix(b_matrix_path, &b_matrix)) {
    printf("Error reading matrix from %s\n", b_matrix_path);
    return -1;
  }
  free(b_matrix_path);

  if (convolve(a_matrix, b_matrix, &output_matrix)) {
    printf("convolve returned a non-zero integer\n");
    return -1;
  }

  char *output_matrix_path = get_output_matrix_path(task);
  if (write_matrix(output_matrix_path, output_matrix)) {
    printf("Error writing matrix to %s\n", output_matrix_path);
    return -1;
  }
  free(output_matrix_path);

  free(a_matrix->data);
  free(b_matrix->data);
  free(output_matrix->data);
  free(a_matrix);
  free(b_matrix);
  free(output_matrix);
  return 0;
}
