#ifndef NN_H
#define NN_H

#include <math.h>
#include <stdio.h>
#include <stdbool.h>
#include "utils.h"

typedef enum {
    SIGMOID,
    RELU,
    TANH,
}actf_t;

typedef struct {
    size_t *arch;
    size_t arch_layers;

    Mat *ws;
    Row bias;
    Row *as;
    actf_t af;
}NN;

// #define MAT_AT(m, i, j) m.elems[i * m.cols + j]
// #define ROW_AT(r, i) r.elems[i]
// #define ARRAY_LEN(x) sizeof(x) / sizeof(x[0])
#define NN_INPUT_COLS(nn) nn.arch[0]
#define NN_OUTPUT_COLS(nn) nn.arch[nn.arch_layers-1]
#define NN_OUTPUT(nn) nn.as[nn.arch_layers-1]
#define NN_INPUT(nn) nn.as[0]

extern float sigf(float x);
extern float reluf(float x);
extern float tanhf(float x);
extern float rand_float(float min, float max);
extern float act_fun(float x, actf_t af);
extern float dactf(float y, actf_t af);
extern void act_row(Row r, actf_t af);

extern void copy_row_to_mat(Mat m, Row r, size_t row);
extern void print_mat(Mat m);
extern Mat mat_nrows(Mat m, size_t row_from, size_t row_to);
extern void rand_mat(Mat m, float min, float max);
extern void sum_mat_scalar(Mat m, float scalar);

extern Row sum_rows(Row r1, Row r2);
extern void sum_row_scalar(Row r, float scalar);
extern Mat row_as_mat(Row r);
extern void rand_row(Row r, float min, float max);
extern void print_row(Row r, const char *end);
extern Row row_slice(Row r, size_t from, size_t to);

extern NN nn_alloc(size_t *arch, size_t arch_layers, actf_t af);
extern void rand_nn(NN nn, float min, float max);
extern void nn_zeros(NN nn);
extern void nn_forward(NN nn, Row input);
extern float nn_loss(NN nn, Mat t);
extern NN nn_backprop(NN nn, Mat t);
extern void nn_learn(NN nn, NN g, float lr);
extern void nn_train(NN nn, size_t batch_size, Mat t, float lr, float *cost);
extern void free_nn(NN *nn);

#endif
