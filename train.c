#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

#include "util/img.h"
#include "neural/nn.h"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    srand(time(NULL) + rank); // mỗi tiến trình seed khác nhau

    int number_imgs = 10000;

    // Tất cả tiến trình đều đọc ảnh (vì bạn đã chia trong network_train_batch_imgs)
    Img** imgs = csv_to_imgs("./data/mnist_test.csv", number_imgs);

    // Tạo mạng nơ-ron
    NeuralNetwork* net = network_create(784, 300, 10, 0.1);

    double start = MPI_Wtime();
    network_train_batch_imgs(net, imgs, number_imgs, 10);
    double end = MPI_Wtime();

    // Chỉ rank 0 lưu mạng
    if (rank == 0) {
        printf("Training took %.2f seconds.\n", end - start);
        network_save(net, "testing_net");
    }

    imgs_free(imgs, number_imgs);
    network_free(net);

    MPI_Finalize();
    return 0;
}
