#include <omp.h>
#include <x86intrin.h>

#include "compute.h"

// Computes the convolution of two matrices
int convolve(matrix_t *a_matrix, matrix_t *b_matrix, matrix_t **output_matrix) {
  uint32_t a_rows = a_matrix->rows;
    uint32_t a_cols = a_matrix->cols;
    uint32_t b_rows = b_matrix->rows;
    uint32_t b_cols = b_matrix->cols;
    uint32_t size = b_rows * b_cols;

    if(a_rows < b_rows || a_cols < b_cols){
        return -1;
    }

    int temp = 0;

    #pragma omp parallel for 
    for(uint32_t i = 0; i < size / 2; ++i) {
        temp = b_matrix->data[i];
        b_matrix->data[i] = b_matrix->data[size - i - 1];
        b_matrix->data[size - i - 1] = temp;
    }

    uint32_t newRows = a_rows - b_rows + 1;
    uint32_t newCols = a_cols - b_cols + 1;

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
    uint32_t new_aRow, new_bRow;
    int sum = 0;

    #pragma omp parallel for collapse(2) schedule(dynamic, 16)
    for(uint32_t i = 0; i < newRows; ++i) {
        for(uint32_t j = 0; j < newCols; ++j) {
            sum = 0;
            for(uint32_t m = 0; m < b_rows; ++m) {
                new_aRow = (i + m) * a_cols + j;
                new_bRow = m * b_cols;

                // SIMD optimization: 8 elements at a time
                __m256i product, a_vec, b_vec;
                __m256i sum_vec = _mm256_setzero_si256();
                int32_t result[8];
                #pragma omp simd reduction(+:sum)
                for (uint32_t n = 0; n < b_cols / 16 * 16; n += 16) {
                    a_vec = _mm256_loadu_si256((__m256i *)&a_matrix->data[new_aRow + n]);
                    b_vec = _mm256_loadu_si256((__m256i *)&b_matrix->data[new_bRow + n]);
                    product = _mm256_mullo_epi32(a_vec, b_vec);
                    sum_vec = _mm256_add_epi32(sum_vec, product);
                    
                    a_vec = _mm256_loadu_si256((__m256i *)&a_matrix->data[new_aRow + n + 8]);
                    b_vec = _mm256_loadu_si256((__m256i *)&b_matrix->data[new_bRow + n + 8]);
                    product = _mm256_mullo_epi32(a_vec, b_vec);
                    sum_vec = _mm256_add_epi32(sum_vec, product);
                }
                #pragma omp simd reduction(+:sum)
                for (uint32_t n = b_cols / 16 * 16; n < b_cols / 8 * 8; n += 8) {
                    a_vec = _mm256_loadu_si256((__m256i *)&a_matrix->data[new_aRow + n]);
                    b_vec = _mm256_loadu_si256((__m256i *)&b_matrix->data[new_bRow + n]);
                    product = _mm256_mullo_epi32(a_vec, b_vec);
                    sum_vec = _mm256_add_epi32(sum_vec, product);
                }
                _mm256_storeu_si256((__m256i *)result, sum_vec);
                sum += result[0] + result[1] + result[2] + result[3] + result[4] + result[5] + result[6] + result[7];
                // Handle remaining elements (less than 8)
                for (uint32_t n = b_cols / 8 * 8; n < b_cols; ++n) {
                    sum += a_matrix->data[new_aRow + n] * b_matrix->data[new_bRow + n];
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
