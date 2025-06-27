#include "nn.h"
#include <sys/stat.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include "../util/utils.h"
#include "../matrix/ops.h"
#include "../neural/activations.h"
#include "../socket/socket_utils.h"

#define MAXCHAR 1000

// 784, 300, 10
NeuralNetwork* network_create(int input, int hidden, int output, double lr) {
	NeuralNetwork* net = malloc(sizeof(NeuralNetwork));
	net->input = input;
	net->hidden = hidden;
	net->output = output;
	net->learning_rate = lr;
	Matrix* hidden_layer = matrix_create(hidden, input);
	Matrix* output_layer = matrix_create(output, hidden);
	matrix_randomize(hidden_layer, hidden);
	matrix_randomize(output_layer, output);
	net->hidden_weights = hidden_layer;
	net->output_weights = output_layer;
	return net;
}

double network_train(NeuralNetwork* net, Matrix* input, Matrix* output) {
	// Feed forward
	Matrix* hidden_inputs	= dot(net->hidden_weights, input);
	Matrix* hidden_outputs = apply(sigmoid, hidden_inputs);
	Matrix* final_inputs = dot(net->output_weights, hidden_outputs);
	Matrix* final_outputs = apply(sigmoid, final_inputs);

	// Find errors
	Matrix* output_errors = subtract(output, final_outputs);
	double loss = 0.0;
	for (int i = 0; i < output->rows; i++) {
		double diff = output->entries[i][0] - final_outputs->entries[i][0];
		loss += diff * diff;
	}

	Matrix* transposed_mat = transpose(net->output_weights);
	Matrix* hidden_errors = dot(transposed_mat, output_errors);
	matrix_free(transposed_mat);

	// Backpropogate
	// output_weights = add(
	// 		 output_weights, 
	//     scale(
	// 			  net->lr, 
	// 			  dot(
	// 		 			multiply(
	// 						output_errors, 
	// 				  	sigmoidPrime(final_outputs)
	// 					), 
	// 					transpose(hidden_outputs)
	// 				)
	// 		 )
	// )
	Matrix* sigmoid_primed_mat = sigmoidPrime(final_outputs);
	Matrix* multiplied_mat = multiply(output_errors, sigmoid_primed_mat);
	transposed_mat = transpose(hidden_outputs);
	Matrix* dot_mat = dot(multiplied_mat, transposed_mat);
	Matrix* scaled_mat = scale(net->learning_rate, dot_mat);
	Matrix* added_mat = add(net->output_weights, scaled_mat);

	matrix_free(net->output_weights); // Free the old weights before replacing
	net->output_weights = added_mat;

	matrix_free(sigmoid_primed_mat);
	matrix_free(multiplied_mat);
	matrix_free(transposed_mat);
	matrix_free(dot_mat);
	matrix_free(scaled_mat);

	// hidden_weights = add(
	// 	 net->hidden_weights,
	// 	 scale (
	//			net->learning_rate
	//    	dot (
	//				multiply(
	//					hidden_errors,
	//					sigmoidPrime(hidden_outputs)	
	//				)
	//				transpose(inputs)
	//      )
	// 	 )
	// )
	// Reusing variables after freeing memory
	sigmoid_primed_mat = sigmoidPrime(hidden_outputs);
	multiplied_mat = multiply(hidden_errors, sigmoid_primed_mat);
	transposed_mat = transpose(input);
	dot_mat = dot(multiplied_mat, transposed_mat);
	scaled_mat = scale(net->learning_rate, dot_mat);
	added_mat = add(net->hidden_weights, scaled_mat);
	matrix_free(net->hidden_weights); // Free the old hidden_weights before replacement
	net->hidden_weights = added_mat; 

	matrix_free(sigmoid_primed_mat);
	matrix_free(multiplied_mat);
	matrix_free(transposed_mat);
	matrix_free(dot_mat);
	matrix_free(scaled_mat);

	// Free matrices
	matrix_free(hidden_inputs);
	matrix_free(hidden_outputs);
	matrix_free(final_inputs);
	matrix_free(final_outputs);
	matrix_free(output_errors);
	matrix_free(hidden_errors);
	return loss;
}

double* network_get_weights(NeuralNetwork* net, int* count_out) {
    int count = 0;

    // Tính tổng số trọng số
    count += net->hidden_weights->rows * net->hidden_weights->cols;
    count += net->output_weights->rows * net->output_weights->cols;

    double* all_weights = (double*)malloc(sizeof(double) * count);
    int idx = 0;

    // Lưu trọng số hidden
    for (int i = 0; i < net->hidden_weights->rows; i++) {
        for (int j = 0; j < net->hidden_weights->cols; j++) {
            all_weights[idx++] = net->hidden_weights->entries[i][j];
        }
    }

    // Lưu trọng số output
    for (int i = 0; i < net->output_weights->rows; i++) {
        for (int j = 0; j < net->output_weights->cols; j++) {
            all_weights[idx++] = net->output_weights->entries[i][j];
        }
    }

    *count_out = count;
    return all_weights;
}

void network_set_weights(NeuralNetwork* net, const double* weights, int count) {
    int idx = 0;

    // Gán lại hidden_weights
    for (int i = 0; i < net->hidden_weights->rows; i++) {
        for (int j = 0; j < net->hidden_weights->cols; j++) {
            net->hidden_weights->entries[i][j] = weights[idx++];
        }
    }

    // Gán lại output_weights
    for (int i = 0; i < net->output_weights->rows; i++) {
        for (int j = 0; j < net->output_weights->cols; j++) {
            net->output_weights->entries[i][j] = weights[idx++];
        }
    }

    if (idx != count) {
        fprintf(stderr, "Warning: mismatch in set_weights (%d vs %d)\n", idx, count);
    }
}

void network_train_batch_imgs(NeuralNetwork* net, Img** imgs, int batch_size, int epochs) {
	double loss_sum = 0.0;
    int loss_count = 0;
    for (int epoch = 0; epoch < epochs; epoch++) {
        //double total_loss = 0.0;
        for (int i = 0; i < batch_size; i++) {
            if (i % 100 == 0) printf("Img No. %d\n", i);

            Img* cur_img = imgs[i];
            Matrix* img_data = matrix_flatten(cur_img->img_data, 0); // 0 = flatten to column vector
            Matrix* output = matrix_create(10, 1);
            output->entries[cur_img->label][0] = 1; // Setting the result

            // Train on this image
			double loss = network_train(net, img_data, output); // New version returns loss
			loss_sum += loss;
			loss_count++;

		if (i % 1000 == 0 && loss_count > 0) {
            printf("Average loss after %d images: %.6f\n", i+1, loss_sum / loss_count);
            loss_sum = 0;
            loss_count = 0;
        }

            // Clean up
            matrix_free(output);
            matrix_free(img_data);
        }

        // In ra loss mỗi epoch
        printf("Epoch %d/%d \n", epoch + 1, epochs);
    }
}

void network_train_batch_imgs_socket(
    NeuralNetwork* net,
    Img** imgs,
    int batch_size,
    int epochs,
    bool is_master,
    const char* ip,
    int port,
    int local_cpu_count, // <--- THÊM THAM SỐ NÀY
	Img** test_imgs
) {
    int weight_count;
    double* weights_buffer = NULL;
    int start_index = 0, end_index = batch_size;

	double loss_sum = 0.0;
    int loss_count = 0;

    // Thiết lập socket
    int sockfd;
    int slaver_cpu_count = 0;

    if (is_master) {
        sockfd = setup_server(port);
        printf("[Master] Server started. Waiting for connection...\n");
        sockfd = accept_client(sockfd);
        printf("[Master] Connection accepted.\n");

        // Nhận số core từ slaver
        recv_all(sockfd, &slaver_cpu_count, sizeof(int));
        printf("[Master] Slaver has %d cores\n", slaver_cpu_count);

        int total_cpu = local_cpu_count + slaver_cpu_count;

        // Phân chia dữ liệu theo tỷ lệ số core
        int master_imgs = (batch_size * local_cpu_count) / total_cpu;
        start_index = 0;
        end_index = master_imgs;

    } else {
        sockfd = connect_to_server(ip, port);
        printf("[Slaver] Connected to master.\n");

        // Gửi số core của slaver cho master
        send_all(sockfd, &local_cpu_count, sizeof(int));

        int master_imgs; // sẽ được tính lại cùng công thức master dùng
        recv_all(sockfd, &master_imgs, sizeof(int));  // Nhận từ master

        start_index = master_imgs;
        end_index = batch_size;
    }

    // Master gửi lại cho slaver số ảnh nó xử lý để slaver biết phần còn lại
    if (is_master) {
        send_all(sockfd, &end_index, sizeof(int)); // end_index chính là số ảnh master xử lý
    }

    printf("[%s] Processing images from %d to %d\n", is_master ? "Master" : "Slaver", start_index, end_index);
	fflush(stdout);

    weights_buffer = network_get_weights(net, &weight_count); // để biết weight_count

    for (int epoch = 0; epoch < epochs; epoch++) {
        for (int i = start_index; i < end_index; i++) {
            // Train
            Img* cur_img = imgs[i];
            Matrix* input = matrix_flatten(cur_img->img_data, 0);
            Matrix* output = matrix_create(10, 1);
            output->entries[cur_img->label][0] = 1;

			double loss = network_train(net, input, output); // New version returns loss
			loss_sum += loss;
			loss_count++;

			if (i % 1000 == 0 && loss_count > 0) {
				printf("*** Average loss after %d images: %.6f ***\n", i, loss_sum / loss_count);
				fflush(stdout);
				loss_sum = 0;
				loss_count = 0;
			}

			if (i % 1000 == 0) {
				double acc = network_predict_imgs(net, test_imgs, 1000);
				printf("*** After %d images: %.2f%% accuracy ***\n", i, acc * 100);
				fflush(stdout);
			}

            matrix_free(input);
            matrix_free(output);

            // Mỗi 1000 ảnh thì trao đổi trọng số
            if ((((i - start_index + 1) % 1000) == 0) || (i == (end_index - 1))) {
                if (weights_buffer) free(weights_buffer);
                weights_buffer = network_get_weights(net, &weight_count);
				
				double t_start, t_end;
				t_start = time_in_seconds();

                if (is_master) {
                    // Nhận trọng số từ slaver
					double t_recv_start, t_recv_end;
					double t_avg_start, t_avg_end;
					double t_send_start, t_send_end;
					
                    double* slave_weights = (double*)malloc(sizeof(double) * weight_count);
					
					t_recv_start = time_in_seconds();
                    recv_all(sockfd, slave_weights, sizeof(double) * weight_count);
					t_recv_end = time_in_seconds();
					
                    printf("[Master] Received weights from slaver at img %d\n", i);
					fflush(stdout);

                    // Trung bình
					t_avg_start = time_in_seconds();
                    for (int j = 0; j < weight_count; j++) {
                        weights_buffer[j] = (weights_buffer[j] + slave_weights[j]) / 2.0;
                    }
					t_avg_end = time_in_seconds();
                    free(slave_weights);

                    // Gửi lại trọng số mới
					t_send_start = time_in_seconds();
                    send_all(sockfd, weights_buffer, sizeof(double) * weight_count);
					t_send_end = time_in_seconds();
					
                    printf("[Master] Sent averaged weights to slaver\n");
					fflush(stdout);

					printf("[Master] recv: %.6f, averaging: %.6f, send: %.6f seconds at img %d\n", 
						   t_recv_end - t_recv_start, t_avg_end - t_avg_start, t_send_end - t_send_start,i);
					fflush(stdout);

                    network_set_weights(net, weights_buffer, weight_count);

                } else {
                    // Gửi trọng số cho master
					double t_send_start, t_send_end;
					double t_recv_start, t_recv_end;
					
					t_send_start = time_in_seconds();
                    send_all(sockfd, weights_buffer, sizeof(double) * weight_count);
					t_send_end = time_in_seconds();
					
                    printf("[Slaver] Sent weights to master at img %d\n", i);
					fflush(stdout);

                    // Nhận lại trọng số đã trung bình
					t_recv_start = time_in_seconds();
                    recv_all(sockfd, weights_buffer, sizeof(double) * weight_count);
					t_recv_end = time_in_seconds();
					
                    printf("[Slaver] Received updated weights from master\n");
					fflush(stdout);

					printf("[Slaver] send: %.6f, recv: %.6f seconds at img %d\n", 
						   t_send_end - t_send_start, t_recv_end - t_recv_start,i);
					fflush(stdout);
					
					network_set_weights(net, weights_buffer, weight_count);
                }

				t_end = time_in_seconds();
				if (is_master) {
					printf("[Master] total sync took %.6f seconds at img %d\n", t_end - t_start, i);
				} else {
					printf("[Slaver] total sync took %.6f seconds at img %d\n", t_end - t_start, i);
				}
				fflush(stdout);
            }
        }

        printf("[%s] Epoch %d/%d done.\n", is_master ? "Master" : "Slaver", epoch + 1, epochs);
        fflush(stdout);
    }

    free(weights_buffer);
    socket_close(sockfd);
}

Matrix* network_predict_img(NeuralNetwork* net, Img* img) {
	Matrix* img_data = matrix_flatten(img->img_data, 0);
	Matrix* res = network_predict(net, img_data);
	matrix_free(img_data);
	return res;
}

double network_predict_imgs(NeuralNetwork* net, Img** imgs, int n) {
	int n_correct = 0;
	for (int i = 0; i < n; i++) {
		Matrix* prediction = network_predict_img(net, imgs[i]);
		if (matrix_argmax(prediction) == imgs[i]->label) {
			n_correct++;
		}
		matrix_free(prediction);
	}
	return 1.0 * n_correct / n;
}

Matrix* network_predict(NeuralNetwork* net, Matrix* input_data) {
	Matrix* hidden_inputs	= dot(net->hidden_weights, input_data);
	Matrix* hidden_outputs = apply(sigmoid, hidden_inputs);
	Matrix* final_inputs = dot(net->output_weights, hidden_outputs);
	Matrix* final_outputs = apply(sigmoid, final_inputs);
	Matrix* result = softmax(final_outputs);

	matrix_free(hidden_inputs);
	matrix_free(hidden_outputs);
	matrix_free(final_inputs);
	matrix_free(final_outputs);

	return result;
}

void network_save(NeuralNetwork* net, char* file_string) {
	mkdir(file_string, 0777);
	// Write the descriptor file
	chdir(file_string);
	FILE* descriptor = fopen("descriptor", "w");
	fprintf(descriptor, "%d\n", net->input);
	fprintf(descriptor, "%d\n", net->hidden);
	fprintf(descriptor, "%d\n", net->output);
	fclose(descriptor);
	matrix_save(net->hidden_weights, "hidden");
	matrix_save(net->output_weights, "output");
	printf("Successfully written to '%s'\n", file_string);
	chdir("-"); // Go back to the orignal directory
}

NeuralNetwork* network_load(char* file_string) {
	NeuralNetwork* net = malloc(sizeof(NeuralNetwork));
	char entry[MAXCHAR];
	chdir(file_string);

	FILE* descriptor = fopen("descriptor", "r");
	fgets(entry, MAXCHAR, descriptor);
	net->input = atoi(entry);
	fgets(entry, MAXCHAR, descriptor);
	net->hidden = atoi(entry);
	fgets(entry, MAXCHAR, descriptor);
	net->output = atoi(entry);
	fclose(descriptor);
	net->hidden_weights = matrix_load("hidden");
	net->output_weights = matrix_load("output");
	printf("Successfully loaded network from '%s'\n", file_string);
	chdir("-"); // Go back to the original directory
	return net;
}

void network_print(NeuralNetwork* net) {
	printf("# of Inputs: %d\n", net->input);
	printf("# of Hidden: %d\n", net->hidden);
	printf("# of Output: %d\n", net->output);
	printf("Hidden Weights: \n");
	matrix_print(net->hidden_weights);
	printf("Output Weights: \n");
	matrix_print(net->output_weights);
}

void network_free(NeuralNetwork *net) {
	matrix_free(net->hidden_weights);
	matrix_free(net->output_weights);
	free(net);
	net = NULL;
}