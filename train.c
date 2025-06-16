#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "util/img.h"
#include "neural/nn.h"

int main() {
    srand(time(NULL));

    int number_imgs = 60000;
    Img** imgs = csv_to_imgs("./data/mnist_train.csv", number_imgs);
    Img** test_imgs = csv_to_imgs("./data/mnist_test.csv", 1000);
    NeuralNetwork* net = network_create(784, 300, 10, 0.1);

    double untrained_acc = network_predict_imgs(net, test_imgs, 1000);
    printf("Untrained network: %.2f%%\n", untrained_acc * 100);

    time_t start = time(NULL);
    network_train_batch_imgs(net, imgs, number_imgs,test_imgs);
    time_t end = time(NULL);

    printf("Training took %.2f seconds.\n", difftime(end, start));
    fflush(stdout);

    network_save(net, "testing_net");

    printf("\n=== Detailed predictions after training ===\n");
double final_acc = network_predict_imgs(net, test_imgs, 1000);
printf("Final test accuracy: %.2f%%\n", final_acc * 100);

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

// ✅ Giải phóng ở đây — sau khi xong hết
imgs_free(imgs, number_imgs);
imgs_free(test_imgs, 1000);
network_free(net);
return 0;
}