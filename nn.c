#include "nn.h"

float sigf(float x){
    return 1 / (1 + expf(-x));
}

float reluf(float x){
    return x > 0 ? x : 0;
}

float tanhf(float x){
    return (expf(x) - expf(-x)) / (expf(x) + expf(-x));
}

float rand_float(float min, float max){
    float n = (float) rand() / (float) RAND_MAX;
    return min + n * (max - min);
}

float act_fun(float x, actf_t af){
    switch (af) {
        case SIGMOID: return sigf(x);
        case RELU: return reluf(x);
        case TANH: return tanh(x);
        default: assert(0 == "unreachable");
    }
}

float dactf(float y, actf_t af){
    switch (af) {
        case SIGMOID: return y * (1 - y);
        case RELU: return y == 0 ? 0 : 1;
        case TANH: return 1 - y * y;
        default: assert(0 == "unreachable");
    }
}

void act_row(Row r, actf_t af){
    for(size_t i = 0; i < r.cols; i++){
        ROW_AT(r, i) = act_fun(ROW_AT(r, i), af);
    }
}




void copy_row_to_mat(Mat m, Row r, size_t row){
    assert(m.cols >= r.cols);
    for(size_t i = 0; i < r.cols; i++){
        MAT_AT(m, row, i) = ROW_AT(r, i);
    }
}

void print_mat(Mat m){
    for(size_t i = 0; i < m.rows; i++){
        for(size_t j = 0; j < m.cols; ++j){
            printf("% f ", MAT_AT(m, i, j));
        }
        printf("\n");
    }
}

//[from, to]
Mat mat_nrows(Mat m, size_t row_from, size_t row_to){
    assert(row_from <= row_to && row_to < m.rows);
    return (Mat){
        .cols = m.cols,
        .rows = row_to - row_from + 1,
        .elems = &MAT_AT(m, row_from, 0),
    };
}

void rand_mat(Mat m, float min, float max){
    for(size_t i = 0; i < m.rows; ++i){
        for(size_t j = 0; j < m.cols; ++j){
            MAT_AT(m, i, j) =  rand_float(min, max);
        }
    }
}

void sum_mat_scalar(Mat m, float scalar){
    for(size_t i = 0; i < m.rows; i++){
        for(size_t j = 0; j < m.cols; j++){
            MAT_AT(m, i, j) += scalar;
        }
    }
}





Row sum_rows(Row r1, Row r2){
    assert(r1.cols == r2.cols);
    Row res = row_alloc(r1.cols);
    for(size_t i = 0; i < res.cols; i++){
        ROW_AT(res, i) = ROW_AT(r1, i) + ROW_AT(r2, i);
    }
    return res;
}

void sum_row_scalar(Row r, float scalar){
    for(size_t i = 0; i < r.cols; i++){
        ROW_AT(r, i) += scalar;
    }
}

/// creates a matrix of one row with the same pointer as the row @r,
/// after this function is used the changes to the returned matrix
/// will affect r too.
Mat row_as_mat(Row r){
    return (Mat) {
        .cols = r.cols,
        .rows = 1,
        .elems = r.elems,
    };
}

// copy r1 -> r2
void copy_row(Row r1, Row r2){
    assert(r1.cols == r2.cols);
    for(size_t i = 0; i < r1.cols; ++i){
        ROW_AT(r2, i) = ROW_AT(r1, i);
    }
}

void rand_row(Row r, float min, float max){
    for(size_t i = 0; i < r.cols; i++){
        ROW_AT(r, i) = rand_float(min, max);
    }
}

void print_row(Row r, const char *end){
    printf("[");
    for(size_t i = 0; i < r.cols - 1; i++){
        printf("%f, ", ROW_AT(r, i));
    }
    printf("%f", ROW_AT(r, r.cols-1));
    printf("] %s", end);
}





NN nn_alloc(size_t *arch, size_t arch_layers, actf_t af){
    NN nn;
    nn.arch = arch;
    nn.arch_layers = arch_layers;
    nn.af = af;
    nn.ws = (Mat *) malloc((arch_layers - 1) * sizeof(Mat));
    nn.as = (Row *) malloc(arch_layers * sizeof(Row));
    assert(nn.ws != NULL && nn.as != NULL);

    nn.bias = row_alloc(arch_layers - 1);
    nn.as[0] = row_alloc(arch[0]);
    for(size_t i = 1; i < arch_layers; i++){
        nn.ws[i-1] = mat_alloc(arch[i-1], arch[i]);
        nn.as[i] = row_alloc(arch[i]);
    }
    return nn;
}

void rand_nn(NN nn, float min, float max){
    rand_row(nn.bias, min, max);
    for(size_t i = 0; i < nn.arch_layers - 1; i++){
        rand_mat(nn.ws[i], min, max);
    }
}

void nn_zeros(NN nn){
    row_fill(nn.as[0], 0.0f);
    row_fill(nn.bias, 0.0f);
    for(size_t i = 1; i < nn.arch_layers; i++){
        mat_fill(nn.ws[i-1], 0.0f);
        row_fill(nn.as[i], 0.0f);
    }
}

void nn_forward(NN nn, Row input){
    copy_row(input, nn.as[0]);
    for(size_t i = 0; i < nn.arch_layers - 1; i++){
        mul_mat(row_as_mat(nn.as[i]), nn.ws[i], row_as_mat(nn.as[i+1]));
        sum_row_scalar(nn.as[i+1], ROW_AT(nn.bias, i));
        act_row(nn.as[i+1], nn.af);
    }
}

float nn_loss(NN nn, Mat t){
    assert(NN_INPUT_COLS(nn) + NN_OUTPUT_COLS(nn) == t.cols);
    float cost = 0.0f;

    for(size_t i = 0; i < t.rows; i++){
        Row in = row_slice(mat_row(t, i), 0, NN_INPUT_COLS(nn) - 1);
        Row y = row_slice(mat_row(t, i), NN_INPUT_COLS(nn), t.cols - 1);
        nn_forward(nn, in);

        for(size_t j = 0; j < NN_OUTPUT_COLS(nn); j++){
            float d = ROW_AT(NN_OUTPUT(nn), j) - ROW_AT(y, j);
            cost += d * d;
        }
    }
    return cost / (float) t.rows;
}

// each row of mat x is an input
// each row of mat t is the expected output
NN nn_backprop(NN nn, Mat t){
    assert(NN_INPUT_COLS(nn) + NN_OUTPUT_COLS(nn) == t.cols);
    NN g = nn_alloc(nn.arch, nn.arch_layers, nn.af);
    nn_zeros(g);

    for(size_t i = 0; i < t.rows; i++){
        Row in = row_slice(mat_row(t, i), 0, NN_INPUT_COLS(nn) - 1);
        Row y = row_slice(mat_row(t, i), NN_INPUT_COLS(nn), t.cols - 1);

        nn_forward(nn, in);

        for(size_t l = 0; l < g.arch_layers; l++){
            row_fill(g.as[l], 0.0f);
        }

        for(size_t j = 0; j < NN_OUTPUT_COLS(nn); j++){
            ROW_AT(NN_OUTPUT(g), j) = 2 * (ROW_AT(NN_OUTPUT(nn), j) - ROW_AT(y, j));
        }

        for(size_t l = nn.arch_layers - 1; l > 0; l--){
            for(size_t j = 0; j < nn.arch[l]; j++){
                float de = ROW_AT(g.as[l], j);
                float da = dactf(ROW_AT(nn.as[l], j), nn.af);
                ROW_AT(g.bias, l-1) += da * de;
                for(size_t k = 0; k < nn.arch[l-1]; k++){
                    float a = ROW_AT(nn.as[l-1], k);
                    float w = MAT_AT(nn.ws[l-1], k, j);
                    ROW_AT(g.as[l-1], k) += w * da * de;
                    MAT_AT(g.ws[l-1], k, j) += a * da * de;
                }
            }
        }
    }

    for(size_t l = 0; l < g.arch_layers - 1; l++){
        for(size_t i = 0; i < g.arch[l]; i++){
            for(size_t j = 0; j < g.arch[l+1]; j++){
                MAT_AT(g.ws[l], i, j) /= t.rows;
            }
        }
        ROW_AT(g.bias, l) /= t.rows;
    }

    return g;
}

void nn_learn(NN nn, NN g, float lr){
    assert(nn.arch_layers == g.arch_layers);
    for(size_t l = 0; l < nn.arch_layers - 1; l++){
        for(size_t i = 0; i < nn.arch[l]; i++){
            for(size_t j = 0; j < nn.arch[l+1]; ++j){
                MAT_AT(nn.ws[l], i, j) -= lr * MAT_AT(g.ws[l], i, j);
            }
        }
        ROW_AT(nn.bias, l) -= lr * ROW_AT(g.bias, l);
    }
}

void nn_train(NN nn, size_t batch_size, Mat t, float lr, float *cost){
    assert(NN_INPUT_COLS(nn) + NN_OUTPUT_COLS(nn) == t.cols && cos != NULL);
    for(size_t i = 0; i < t.rows; i += batch_size){
        size_t to = i + batch_size - 1 > t.rows ? t.rows - 1: i + batch_size - 1;
        Mat batch = mat_nrows(t, i, to);
        // printf("%zu, %zu, %zu\n", t.rows, i, to);
        // print_mat(batch);
        // printf("\n");

        NN g = nn_backprop(nn, batch);
        nn_learn(nn, g, lr);
        *cost = nn_loss(nn, batch);
        free_nn(&g);
    }
}

void free_nn(NN *nn){
    free_row(&nn->as[0]);
    for(size_t i = 1; i < nn->arch_layers; i++){
        free_mat(&nn->ws[i-1]);
        free_row(&nn->as[i]);
    }
    free_row(&nn->bias);
    nn->arch_layers = 0;
    nn->arch = NULL;
    free(nn->ws);
    free(nn->as);
}
