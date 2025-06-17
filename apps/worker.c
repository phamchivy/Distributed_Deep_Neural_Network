#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "../neural/nn.h"
#include "../util/img.h"
#include "../socket/socket_utils.h"

void sync_with_parameter_server(NeuralNetwork* net, int worker_id, 
                               const char* server_ip, int server_port, int sync_count);

int main(int argc, char** argv) {
    if (argc < 4) {
        printf("Usage: %s <worker_id> <server_ip> <server_port>\n", argv[0]);
        return 1;
    }
    
    int worker_id = atoi(argv[1]);
    const char* server_ip = argv[2];
    int server_port = atoi(argv[3]);
    
    printf("[Worker %d] Starting, server: %s:%d\n", worker_id, server_ip, server_port);
    fflush(stdout);
    
    srand(time(NULL) + worker_id);  // Different seed per worker
    
    // Wait a bit for parameter server to start
    sleep(2);
    
    // Load data
    printf("[Worker %d] Loading MNIST data...\n", worker_id);
    int number_imgs = 60000;
    Img** imgs = csv_to_imgs("./data/mnist_train.csv", number_imgs);
    Img** test_imgs = csv_to_imgs("./data/mnist_test.csv", 1000);
    
    // Data partitioning (50-50 split)
    int start_index = (worker_id - 1) * (number_imgs / 2);
    int end_index = worker_id * (number_imgs / 2);
    
    printf("[Worker %d] Processing images %d to %d (%d total)\n", 
           worker_id, start_index, end_index-1, end_index - start_index);
    fflush(stdout);
    
    // Initialize network
    NeuralNetwork* net = network_create(784, 300, 10, 0.1);
    network_easgd_init(net, 0.01, 0.001);  // α=0.1, β=0.01
    
    printf("[Worker %d] Network initialized with EASGD: α=0.01, β=0.001\n", worker_id);
    fflush(stdout);
    
    // Training loop
    double loss_sum = 0.0;
    int loss_count = 0;
    int sync_count = 0;
    
    double start_time = time_in_socket_seconds();
    
    for (int epoch = 0; epoch < 1; epoch++) {
        printf("[Worker %d] Starting epoch %d\n", worker_id, epoch + 1);
        fflush(stdout);
        
        for (int i = start_index; i < end_index; i++) {
            // Standard training
            Img* cur_img = imgs[i];
            Matrix* input = matrix_flatten(cur_img->img_data, 0);
            Matrix* output = matrix_create(10, 1);
            output->entries[cur_img->label][0] = 1;

            double loss = network_train(net, input, output);
            loss_sum += loss;
            loss_count++;

            // Async sync every 1000 images
            if ((i - start_index + 1) % 1000 == 0) {
                sync_with_parameter_server(net, worker_id, server_ip, server_port, ++sync_count);
                
                // Test accuracy after sync
                double acc = network_predict_imgs(net, test_imgs, 1000);
                printf("[Worker %d] After %d images: accuracy=%.2f%%\n", 
                       worker_id, i - start_index + 1, acc * 100);
                fflush(stdout);
            }

            // Logging
            if ((i - start_index + 1) % 1000 == 0 && loss_count > 0) {
                printf("[Worker %d] After %d images: avg_loss=%.6f\n", 
                       worker_id, i - start_index + 1, loss_sum / loss_count);
                fflush(stdout);
                loss_sum = 0;
                loss_count = 0;
            }

            matrix_free(input);
            matrix_free(output);
        }
        
        printf("[Worker %d] Epoch %d completed\n", worker_id, epoch + 1);
        fflush(stdout);
    }
    
    double end_time = time_in_socket_seconds();
    printf("[Worker %d] Training completed in %.2f seconds, total syncs: %d\n", 
           worker_id, end_time - start_time, sync_count);
    fflush(stdout);
    
    // Save final model
    char model_name[256];
    sprintf(model_name, "worker_%d_final", worker_id);
    network_save(net, model_name);
    
    // Final test
    double final_acc = network_predict_imgs(net, test_imgs, 1000);
    printf("[Worker %d] Final test accuracy: %.2f%%\n", worker_id, final_acc * 100);
    fflush(stdout);
    
    // Cleanup
    imgs_free(imgs, number_imgs);
    imgs_free(test_imgs, 1000);
    network_free(net);
    
    return 0;
}

void sync_with_parameter_server(NeuralNetwork* net, int worker_id, 
                               const char* server_ip, int server_port, int sync_count) {
    double sync_start = time_in_socket_seconds();
    double comm_start_1, comm_end_1;
    double comm_start_2, comm_end_2;
    
    // Connect to parameter server
    int sockfd = connect_to_server(server_ip, server_port);
    if (sockfd < 0) {
        printf("[Worker %d] Failed to connect to parameter server\n", worker_id);
        return;
    }
    
    // Send worker ID
    if (send_all(sockfd, &worker_id, sizeof(int)) != sizeof(int)) {
        printf("[Worker %d] Failed to send worker ID\n", worker_id);
        close(sockfd);
        return;
    }
    
    // Send weights
    int weight_count;
    double* weights = network_get_weights(net, &weight_count);


    // === TIMING: Send weights ===
    comm_start_1 = time_in_socket_seconds();
    if (send_all(sockfd, &weight_count, sizeof(int)) != sizeof(int) ||
        send_all(sockfd, weights, sizeof(double) * weight_count) != sizeof(double) * weight_count) {
        printf("[Worker %d] Failed to send weights\n", worker_id);
        free(weights);
        close(sockfd);
        return;
    }
    comm_end_1 = time_in_socket_seconds();
    printf("[Worker %d] Sent %d weights in %.3fms\n", worker_id, weight_count, (comm_end_1 - comm_start_1)*1000);
    fflush(stdout);
    
    // Receive elastic center
    int center_count;
    if (recv_all(sockfd, &center_count, sizeof(int)) != sizeof(int)) {
        printf("[Worker %d] Failed to receive center count\n", worker_id);
        free(weights);
        close(sockfd);
        return;
    }
    
    // === TIMING: Receive center weights ===
    comm_start_1 = time_in_socket_seconds();
    double* center_weights = malloc(sizeof(double) * center_count);
    if (recv_all(sockfd, center_weights, sizeof(double) * center_count) != sizeof(double) * center_count) {
        printf("[Worker %d] Failed to receive center weights\n", worker_id);
        free(weights);
        free(center_weights);
        close(sockfd);
        return;
    }
    comm_end_1 = time_in_socket_seconds();
    printf("[Worker %d] Received %d center weights in %.3fms\n", 
           worker_id, center_count, (comm_end_1 - comm_start_1)*1000);
    fflush(stdout);
    
    // Apply elastic averaging: wᵢ ← wᵢ + α(w̄ - wᵢ)
    network_apply_elastic_averaging(net, center_weights, center_count);
    
    double sync_end = time_in_socket_seconds();
    printf("[Worker %d] Sync #%d completed in %.3fms (total), comm overhead: %.1f%%\n", 
           worker_id, sync_count, (sync_end - sync_start)*1000,
           (((comm_end_2 - comm_start_2) + (comm_end_1 - comm_start_1)) / (sync_end - sync_start)) * 100);
    fflush(stdout);
    
    // Cleanup
    free(weights);
    free(center_weights);
    close(sockfd);
}