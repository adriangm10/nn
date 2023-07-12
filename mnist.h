#ifndef MNIST_H
#define MNIST_H

#include "utils.h"
#include <fcntl.h>
#include <unistd.h>

#define TRAIN_IMGS "./data/train-images-idx3-ubyte"
#define TRAIN_LBLS "./data/train-labels-idx1-ubyte"
#define TEST_IMGS "./data/t10k-images-idx3-ubyte"
#define TEST_LBLS "./data/t10k-labels-idx1-ubyte"
#define IMG_SIZE 784  //28*28
#define IMG_SIDE 28
#define MAX_TRAIN 60000
#define MAX_TEST 10000

extern int reverse_int(int i);
extern void read_labels(char *file_name, Mat t, int from, int *to);
extern void read_imgs(char *file_name, Mat t, int from, int *to);
extern Mat charge_mnist(int from, int *to);
extern Mat charge_mnist_test(int from, int *to);

#endif
