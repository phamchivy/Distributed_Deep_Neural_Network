#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "util/img.h"
#include "neural/nn.h"

int main() {
    srand(time(NULL));

    int number_imgs = 10000;
    Img** imgs = csv_to_imgs("./data/mnist_test.csv", number_imgs);
    NeuralNetwork* net = network_create(784, 300, 10, 0.1);

    time_t start = time(NULL);
    network_train_batch_imgs(net, imgs, number_imgs,10);
    time_t end = time(NULL);

    printf("Training took %.2f seconds.\n", difftime(end, start));
    fflush(stdout);

    network_save(net, "testing_net");

    imgs_free(imgs, number_imgs);
    network_free(net);
    return 0;
}