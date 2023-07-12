#include "utils.h"

Mat mat_alloc(size_t rows, size_t cols){
    Mat m;
    m.elems = (float *) malloc(cols * rows * sizeof(float));
    assert(m.elems != NULL);
    m.cols = cols;
    m.rows = rows;
    return m;
}

void mul_mat(Mat m1, Mat m2, Mat out){
    assert(m1.cols == m2.rows);
    assert(m1.rows == out.rows && m2.cols == out.cols);

    for(size_t i = 0; i < m1.rows; i++){
        for(size_t j = 0; j < m2.cols; j++){
            MAT_AT(out, i, j) = 0.0f;
            for(size_t k = 0; k < m1.cols; k++){
                MAT_AT(out, i, j) += MAT_AT(m1, i, k) * MAT_AT(m2, k, j);
            }
        }
    }
}

void mat_fill(Mat m, float x){
    for(size_t i = 0; i < m.rows; ++i){
        for(size_t j = 0; j < m.cols; ++j){
            MAT_AT(m, i, j) = x;
        }
    }
}

void free_mat(Mat *m){
    m->rows = 0;
    m->cols = 0;
    free(m->elems);
    m->elems = NULL;
}

Row row_alloc(size_t cols){
    Row row;
    row.cols = cols;
    row.elems = (float *) malloc(cols * sizeof(float));
    assert(row.elems != NULL);
    return row;
}

void row_fill(Row r, float x){
    for(size_t i = 0; i < r.cols; i++){
        ROW_AT(r, i) = x;
    }
}

///[from, to]
Row row_slice(Row r, size_t from, size_t to){
    assert(to >= from && to < r.cols);
    return (Row) {
        .elems = &ROW_AT(r, from),
        .cols = to - from + 1,
    };
}

void free_row(Row *r){
    r->cols = 0;
    free(r->elems);
    r->elems = NULL;
}

