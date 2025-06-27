#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "util/utils.h"
#include "util/img.h"
#include "neural/nn.h"

int main(int argc, char** argv) {
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }

    srand(time(NULL));

    const char* role = argv[1];
    int port = atoi(argv[2]);

    int number_imgs = 60000;
    Img** imgs = csv_to_imgs("./data/mnist_train.csv", number_imgs);
    Img** test_imgs = csv_to_imgs("./data/mnist_test.csv", 1000);

    NeuralNetwork* net = network_create(784, 300, 10, 0.1);

    double start = time_in_seconds();

    if (strcmp(role, "master") == 0) {
        printf("[MASTER] Starting on port %d\n", port);
        fflush(stdout);
        log_allowed_cpus();
        int cpu_count = get_allowed_cpu_count();
        printf("[%s] Allowed CPU cores: %d\n", role, cpu_count);
        fflush(stdout);
        network_train_batch_imgs_socket(net, imgs, number_imgs, 1, true, NULL, port,cpu_count,test_imgs);
    } else if (strcmp(role, "slaver") == 0) {
        if (argc < 4) {
            print_usage(argv[0]);
            return 1;
        }
        const char* master_ip = argv[3];
        log_allowed_cpus();
        int cpu_count = get_allowed_cpu_count();
        printf("[%s] Allowed CPU cores: %d\n", role, cpu_count);
        printf("[SLAVER] Connecting to %s:%d\n", master_ip, port);
        fflush(stdout);
        network_train_batch_imgs_socket(net, imgs, number_imgs, 1, false, master_ip, port,cpu_count,test_imgs);
    } else {
        print_usage(argv[0]);
        return 1;
    }

    double end = time_in_seconds();
    printf("[%s] Training took %.2f seconds.\n", role, end - start);

    if (strcmp(role, "master") == 0) {
        network_save(net, "testing_net");
    }

        // Chi tiết 5 predictions
    for (int i = 0; i < 5; i++) {
        Matrix* pred = network_predict_img(net, test_imgs[i]);
        printf("\nImage %d - True label: %d\n", i, test_imgs[i]->label);
        printf("Raw sigmoid outputs:\n");
        double max_val = 0;
        int max_idx = 0;
        for (int j = 0; j < 10; j++) {
            double val = pred->entries[j][0];
            printf("  Class %d: %.4f\n", j, val);
            if (val > max_val) {
                max_val = val;
                max_idx = j;
            }
        }
        printf("Predicted: %d (confidence: %.4f)\n", max_idx, max_val);
        printf("Correct: %s\n", (max_idx == test_imgs[i]->label) ? "YES" : "NO");

        matrix_free(pred);
    }

    imgs_free(imgs, number_imgs);
    imgs_free(test_imgs, 1000);
    network_free(net);

    return 0;
}
