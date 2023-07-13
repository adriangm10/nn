#include "mnist.h"
#include <stdio.h>

int reverse_int(int i){
    unsigned char c1, c2, c3, c4;
    c1 = i & 0xFF;
    c2 = (i >> 8) & 0xFF;
    c3 = (i >> 16) & 0xFF;
    c4 = (i >> 24) & 0xFF;

    return (int) ((c1 << 24) | (c2 << 16) | (c3 << 8) | c4);
}

// [from, to]
// starts counting from 1
// if to = -1, take from @from to the end
void read_labels(char *file_name, Mat t, int from, int *to){
    assert(t.cols == IMG_SIZE + 10 && (from < *to || *to == -1) && from >= 1);
    int magic_number, fd, n;

    fd = open(file_name, O_RDONLY);
    assert(fd != -1 && "could not open the file");

    read(fd, &magic_number, sizeof(magic_number));
    magic_number = reverse_int(magic_number);
    assert(magic_number == 2049 && "not a mnist label file");

    read(fd, &n, sizeof(n));
    n = reverse_int(n);
    assert(from < n);

    if(*to == -1) *to = n;
    *to = *to >= n ? n : *to;
    assert(*to - from <= t.rows);
    lseek(fd, (from - 1) * sizeof(unsigned char), SEEK_CUR);

    for(int i = 0; i < *to - from + 1; i++){
        unsigned char temp;
        read(fd, &temp, sizeof(unsigned char));
        Row lbl_row = row_slice(mat_row(t, i), t.cols - 10, t.cols - 1);
        row_fill(lbl_row, 0.0f);
        ROW_AT(lbl_row, temp) = 1.0f;
    }

    close(fd);
}

// [from, to]
// starts counting from 1
// if to = -1, take from @from to the end
void read_imgs(char *file_name, Mat t, int from, int *to){
    assert(t.cols == IMG_SIZE + 10 && (from < *to || *to == -1) && from >= 1);
    int magic_number, fd, n, rows, cols;

    fd = open(file_name, O_RDONLY);
    assert(fd != -1 && "could not open the file");

    read(fd, &magic_number, sizeof(magic_number));
    magic_number = reverse_int(magic_number);
    assert(magic_number == 2051 && "not a mnist image file");

    read(fd, &n, sizeof(int));
    n = reverse_int(n);
    assert(from < n);

    read(fd, &rows, sizeof(int));
    rows = reverse_int(rows);

    read(fd, &cols, sizeof(int));
    cols = reverse_int(cols);
    assert(rows == cols && rows == IMG_SIDE);

    if(*to == -1) *to = n;
    *to = *to >= n ? n : *to;
    assert(*to - from <= t.rows);
    lseek(fd, (from - 1) * IMG_SIZE * sizeof(unsigned char), SEEK_CUR);

    for(int i = 0; i < *to - from + 1; ++i){
        for(int j = 0; j < rows; ++j){
            for(int k = 0; k < cols; ++k){
                unsigned char temp;
                read(fd, &temp, sizeof(temp));
                MAT_AT(t, i, j * IMG_SIDE + k) = (float) temp;
            }
        }
    }

    close(fd);
}

// [from, to]
// starts counting from 1
// if to = -1, take from @from to the end
Mat charge_mnist(int from, int *to){
    assert(*to >= -1);
    if(*to == -1 || *to > MAX_TRAIN) *to = MAX_TRAIN;
    Mat t = mat_alloc(*to - from + 1, IMG_SIZE + 10);
    read_labels(TRAIN_LBLS, t, from, to);
    read_imgs(TRAIN_IMGS, t, from, to);
    return t;
}

// [from, to]
// starts counting from 1
// if to = -1, take from @from to the end
Mat charge_mnist_test(int from, int *to){
    if(*to == -1 || *to > MAX_TEST) *to = MAX_TEST;
    Mat t = mat_alloc(*to - from + 1, IMG_SIZE + 10);
    read_labels(TEST_LBLS, t, from, to);
    read_imgs(TEST_IMGS, t, from, to);
    return t;
}

// int main(void){
//     int from = 1;
//     int to = 5;
//     Mat t = mat_alloc(to - from, IMG_SIZE + 10);
//     read_imgs(TRAIN_IMGS, t, from, &to);
//     read_labels(TRAIN_LBLS, t, from, &to);
// 
//     for(int k = 0; k < t.rows; k++){
//         for(int i = 0; i < IMG_SIDE; ++i){
//             for(int j = 0; j < IMG_SIDE; ++j){
//                 printf("%3.0f ", MAT_AT(t, k, i * IMG_SIDE + j));
//             }
//             printf("\n");
//         }
//         print_row(row_slice(mat_row(t, k), t.cols - 10, t.cols - 1), "\n");
//     }
// 
//     free_mat(&t);
//     return 0;
// }
