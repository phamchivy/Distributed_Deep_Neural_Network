#include <stdio.h>
#include <stdlib.h>
#include "util/img.h"
#include "neural/nn.h"

double test_random_accuracy(Img** imgs, int n) {
    int correct = 0;
    srand(42); // Fixed seed để reproducible
    for(int i = 0; i < n; i++) {
        int random_guess = rand() % 10;
        if(random_guess == imgs[i]->label) correct++;
    }
    return (double)correct / n;
}

int main() {
    int number_test_imgs = 10000;
    Img** test_imgs = csv_to_imgs("./data/mnist_test.csv", number_test_imgs);
    NeuralNetwork* net = network_load("testing_net_single");

    double random_acc = test_random_accuracy(test_imgs, 10000);
    printf("Random baseline: %.2f%%\n", random_acc * 100);

    // Test: In thử 20 ảnh đầu
    for (int i = 0; i < 20; i++) {
        printf("\n=== Ảnh #%d ===\n", i);
        printf("Nhãn đúng (label): %d\n", test_imgs[i]->label);

        Matrix* prediction = network_predict_img(net, test_imgs[i]);
        int predicted_label = matrix_argmax(prediction);
        printf("Model dự đoán: %d\n", predicted_label);

        if (predicted_label == test_imgs[i]->label) {
            printf("✅ Đúng\n");
        } else {
            printf("❌ Sai\n");
        }

        matrix_free(prediction);

        // Nếu muốn in ra hình ảnh để hình dung (có thể bỏ qua nếu không cần)
    }

    // Đo độ chính xác chung
    double score = network_predict_imgs(net, test_imgs, number_test_imgs);
    printf("\nAccuracy tổng thể: %1.5f\n", score);

    imgs_free(test_imgs, number_test_imgs);
    network_free(net);
    return 0;
}
