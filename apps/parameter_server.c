#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdarg.h> 
#include <unistd.h>  
#include "../neural/nn.h"
#include "../socket/socket_utils.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <port>\n", argv[0]);
        return 1;
    }
    
    int port = atoi(argv[1]);
    printf("[Parameter Server] Starting on port %d\n", port);
    fflush(stdout);
    
    srand(time(NULL));
    
    // Initialize template network và elastic center
    NeuralNetwork* template_net = network_create(784, 300, 10, 0.1);
    network_easgd_init(template_net, 0.01, 0.001);  // α=0.1, β=0.01
    elastic_center_init(template_net);
    
    printf("[Parameter Server] Elastic center initialized\n");
    fflush(stdout);
    
    // Setup server socket
    int server_sock = setup_server(port);
    printf("[Parameter Server] Server socket ready, waiting for workers...\n");
    fflush(stdout);
    
    int request_count = 0;
    
    while (1) {
        // CHỜ CONNECTION (không đo thời gian này)
        printf("[Parameter Server] Waiting for worker connection...\n");
        fflush(stdout);
        
        int client_sock = accept_client(server_sock);
        request_count++;
        
        // BẮT ĐẦU ĐO THỜI GIAN SAU KHI ĐÃ CONNECT
        double request_start = time_in_socket_seconds();
        double comm_start_1, comm_end_1;
        double comm_start_2, comm_end_2;
        
        printf("[Parameter Server] Worker connected (request #%d), starting timer\n", request_count);
        fflush(stdout);
        
        // Receive worker ID
        int worker_id;
        if (recv_all(client_sock, &worker_id, sizeof(int)) != sizeof(int)) {
            printf("[Parameter Server] Failed to receive worker ID\n");
            close(client_sock);
            continue;
        }
        
        // Receive weight count
        int weight_count;
        if (recv_all(client_sock, &weight_count, sizeof(int)) != sizeof(int)) {
            printf("[Parameter Server] Failed to receive weight count\n");
            close(client_sock);
            continue;
        }

        // === TIMING: Receive worker weights ===
        comm_start_1 = time_in_socket_seconds();
        double* worker_weights = malloc(sizeof(double) * weight_count);
        if (recv_all(client_sock, worker_weights, sizeof(double) * weight_count) != sizeof(double) * weight_count) {
            printf("[Parameter Server] Failed to receive weights\n");
            free(worker_weights);
            close(client_sock);
            continue;
        }
        comm_end_1 = time_in_socket_seconds();
        
        printf("[Parameter Server] Received %d weights from worker %d in %.3fms\n", 
               weight_count, worker_id, (comm_end_1 - comm_start_1) * 1000);
        fflush(stdout);
        
        // Update elastic center
        double update_start = time_in_socket_seconds();
        elastic_center_update(template_net, worker_weights, weight_count);
        double update_end = time_in_socket_seconds();

        printf("[Parameter Server] Updated elastic center in %.3fms\n", 
               (update_end - update_start) * 1000);
        fflush(stdout);
        
        // Get center weights
        int center_count;
        double* center_weights = elastic_center_get_weights(&center_count);
        
        // === TIMING: Send center weights ===
        comm_start_2 = time_in_socket_seconds();
        if (send_all(client_sock, &center_count, sizeof(int)) == sizeof(int) &&
            send_all(client_sock, center_weights, sizeof(double) * center_count) == sizeof(double) * center_count) {
            comm_end_2 = time_in_socket_seconds();
            printf("[Parameter Server] Sent %d center weights to worker %d in %.3fms\n", 
                   center_count, worker_id, (comm_end_2 - comm_start_2) * 1000);
        } else {
            printf("[Parameter Server] Failed to send center to worker %d\n", worker_id);
        }
        fflush(stdout);

        double request_end = time_in_socket_seconds();
        
        // TÍNH THỜI GIAN ĐÚNG
        double total_processing_time = (request_end - request_start) * 1000; // ms
        double recv_time = (comm_end_1 - comm_start_1) * 1000;
        double update_time = (update_end - update_start) * 1000;
        double send_time = (comm_end_2 - comm_start_2) * 1000;
        double total_comm_time = recv_time + send_time;
        
        printf("[Parameter Server] Request #%d processing: %.3fms total (recv=%.3fms, update=%.3fms, send=%.3fms)\n", 
               request_count, total_processing_time, recv_time, update_time, send_time);
        printf("[Parameter Server] Communication: %.1f%%, Computation: %.1f%%\n",
               (total_comm_time / total_processing_time) * 100,
               (update_time / total_processing_time) * 100);
        fflush(stdout);

        // Cleanup
        free(worker_weights);
        free(center_weights);
        close(client_sock);
        
        if (request_count % 10 == 0) {
            printf("[Parameter Server] === Processed %d total requests ===\n", request_count);
            fflush(stdout);
        }
    }
    
    return 0;
}