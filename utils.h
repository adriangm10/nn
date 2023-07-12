#ifndef UTILS_H
#define UTILS_H

#include <stddef.h>
#include <stdlib.h>
#include <assert.h>

typedef struct {
    size_t rows;
    size_t cols;
    float *elems;
}Mat;

typedef struct {
    size_t cols;
    float *elems;
}Row;

#define MAT_AT(m, i, j) m.elems[i * m.cols + j]
#define ROW_AT(r, i) r.elems[i]
#define ARRAY_LEN(x) sizeof(x) / sizeof(x[0])

extern Mat mat_alloc(size_t rows, size_t cols);
extern void mul_mat(Mat m1, Mat m2, Mat out);
extern void mat_fill(Mat m, float x);
extern void free_mat(Mat *m);

extern Row row_alloc(size_t cols);
extern void row_fill(Row r, float x);
extern Row row_slice(Row r, size_t from, size_t to);
extern void free_row(Row *r);

#endif
