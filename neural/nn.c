#include "nn.h"
#include <sys/stat.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h> 
#include "../matrix/ops.h"
#include "../neural/activations.h"

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

void network_train(NeuralNetwork* net, Matrix* input, Matrix* output) {
	// Feed forward
	Matrix* hidden_inputs	= dot(net->hidden_weights, input);
	Matrix* hidden_outputs = apply(sigmoid, hidden_inputs);
	Matrix* final_inputs = dot(net->output_weights, hidden_outputs);
	Matrix* final_outputs = apply(sigmoid, final_inputs);

	// Find errors
	Matrix* output_errors = subtract(output, final_outputs);
	Matrix* transposed_mat = transpose(net->output_weights);
	Matrix* hidden_errors = dot(transposed_mat, output_errors);
	matrix_free(transposed_mat);

	// Backpropogate
	// output_weights = add(
	//		 output_weights, 
	//     scale(
	// 			  net->lr, 
	//			  dot(
	// 		 			multiply(
	// 						output_errors, 
	//				  	sigmoidPrime(final_outputs)
	//					), 
	//					transpose(hidden_outputs)
	// 				)
	//		 )
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
}

// Cross-Entropy Loss (cho bài toán phân loại)
double calculate_loss(Matrix* predicted, Matrix* actual) {
    double loss = 0.0;
    for (int i = 0; i < predicted->rows; i++) {
        // Tránh log(0) bằng cách sử dụng giá trị nhỏ nhất
        double pred = predicted->entries[i][0];
        if (pred < 1e-10) pred = 1e-10;  // Tránh log(0)
        loss -= actual->entries[i][0] * log(pred);
    }
    return loss / predicted->rows;
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
    for (int epoch = 0; epoch < epochs; epoch++) {
        double total_loss = 0.0;
        for (int i = 0; i < batch_size; i++) {
            if (i % 100 == 0) printf("Img No. %d\n", i);

            Img* cur_img = imgs[i];
            Matrix* img_data = matrix_flatten(cur_img->img_data, 0); // 0 = flatten to column vector
            Matrix* output = matrix_create(10, 1);
            output->entries[cur_img->label][0] = 1; // Setting the result

            // Train on this image
            network_train(net, img_data, output);

            // Calculate loss
            //double loss = calculate_loss(output, net->output);
            //total_loss += loss;

            // Clean up
            matrix_free(output);
            matrix_free(img_data);
        }

        // In ra loss mỗi epoch
        printf("Epoch %d/%d: Loss = %f\n", epoch + 1, epochs, batch_size);
    }
}

void network_train_batch_imgs_allreduce(NeuralNetwork* net, Img** imgs, int batch_size, int epochs) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int imgs_per_proc = batch_size / size;
    int start_index = rank * imgs_per_proc;
    int end_index = (rank + 1) * imgs_per_proc;

    for (int epoch = 0; epoch < epochs; epoch++) {
        double total_loss = 0.0;

        for (int i = start_index; i < end_index; i++) {
            if ((i - start_index) % 1000 == 0) {
                printf("[Rank %d] Img No. %d\n", rank, i);
            }

            Img* cur_img = imgs[i];
            Matrix* img_data = matrix_flatten(cur_img->img_data, 0);
            Matrix* output = matrix_create(10, 1);
            output->entries[cur_img->label][0] = 1;

            // Train
            network_train(net, img_data, output);

            // Free
            matrix_free(output);
            matrix_free(img_data);

            // Sau mỗi 100 ảnh, đồng bộ hóa trọng số giữa các tiến trình
            if ((i - start_index + 1) % 200 == 0) {
                int weight_count;
                double* local_weights = network_get_weights(net, &weight_count); // bạn cần viết hàm này
                double* avg_weights = (double*)malloc(sizeof(double) * weight_count);

                // Tổng trọng số trên tất cả tiến trình
                MPI_Allreduce(local_weights, avg_weights, weight_count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

                // Trung bình trọng số
                for (int j = 0; j < weight_count; j++) {
                    avg_weights[j] /= size;
                }

                // Cập nhật lại trọng số trung bình cho mạng
                network_set_weights(net, avg_weights, weight_count); // bạn cần viết hàm này

                free(local_weights);
                free(avg_weights);
            }
        }

        // Mỗi tiến trình in loss cho epoch của nó
        printf("[Rank %d] Epoch %d/%d done.\n", rank, epoch + 1, epochs);
    }
}

void network_train_batch_imgs_params(NeuralNetwork* net, Img** imgs, int batch_size, int epochs) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int weight_count = 0;
    double* weights_buffer = NULL;

    // Rank 1 sẽ khởi tạo weight_count một lần rồi broadcast cho các rank khác
    if (rank == 1) {
        double* temp = network_get_weights(net, &weight_count);
        free(temp);
    }

    // Broadcast weight_count cho toàn bộ các tiến trình
    MPI_Bcast(&weight_count, 1, MPI_INT, 1, MPI_COMM_WORLD);

    // Tính số ảnh mỗi rank xử lý (có phần dư)
    int base = batch_size / (size - 1);
    int remainder = batch_size % (size - 1);
    int imgs_per_proc, start_index, end_index;

    if (rank > 0) {
        int extra = (rank - 1 < remainder) ? 1 : 0;
        imgs_per_proc = base + extra;
        start_index = (rank - 1) * base + ((rank - 1 < remainder) ? (rank - 1) : remainder);
        end_index = start_index + imgs_per_proc;
    }

    // Rank 0: Thu thập và trung bình trọng số từ các rank khác
    if (rank == 0) {
        weights_buffer = (double*)malloc(sizeof(double) * weight_count);

        for (int epoch = 0; epoch < epochs; epoch++) {
            int batches = base / 100; // Tổng số batch = tổng ảnh / batch size

            for (int batch = 0; batch < batches; batch++) {
                double* sum_weights = (double*)calloc(weight_count, sizeof(double));

                for (int src = 1; src < size; src++) {
                    MPI_Recv(weights_buffer, weight_count, MPI_DOUBLE, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					printf("[Rank 0] Received weights from Rank %d.\n", src);
					fflush(stdout);
                    for (int i = 0; i < weight_count; i++) {
                        sum_weights[i] += weights_buffer[i];
                    }
                }

                // Trung bình trọng số
                for (int i = 0; i < weight_count; i++) {
                    sum_weights[i] /= (size - 1);
                }

				network_set_weights(net, sum_weights, weight_count);
				printf("[Rank %d] save weight\n", rank);
        		fflush(stdout);

                // Gửi lại trọng số trung bình cho tất cả rank
                for (int dest = 1; dest < size; dest++) {
                    MPI_Send(sum_weights, weight_count, MPI_DOUBLE, dest, 1, MPI_COMM_WORLD);
					printf("[Rank 0] Bcast weights to Rank %d.\n", dest);
					fflush(stdout);
                }

                free(sum_weights);
            }

            printf("[Rank 0] Epoch %d/%d done.\n", epoch + 1, epochs);
            fflush(stdout);
        }

        free(weights_buffer);
    } 
    // Rank > 0: Train mạng neural và gửi trọng số sau mỗi batch
    else {
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = start_index; i < end_index; i++) {
                if ((i - start_index) % 1000 == 0) {
                    printf("[Rank %d] Img No. %d\n", rank, i);
                    fflush(stdout);
                }

                Img* cur_img = imgs[i];
                Matrix* img_data = matrix_flatten(cur_img->img_data, 0);
                Matrix* output = matrix_create(10, 1);
                output->entries[cur_img->label][0] = 1;

                network_train(net, img_data, output);

                matrix_free(output);
                matrix_free(img_data);

                // Gửi trọng số sau mỗi 200 ảnh (1 batch) hoặc cuối tập
                if ((i - start_index + 1) % 100 == 0 || i == end_index - 1) {
                    if (weights_buffer) free(weights_buffer);
                    weights_buffer = network_get_weights(net, &weight_count);
					
					printf("[Rank %d] Sending weights to Rank 0 at image %d\n", rank, i);
					fflush(stdout);
                    MPI_Send(weights_buffer, weight_count, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                    MPI_Recv(weights_buffer, weight_count, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    network_set_weights(net, weights_buffer, weight_count);

                    free(weights_buffer);
                    weights_buffer = NULL;
                }
            }

            printf("[Rank %d] Epoch %d/%d done.\n", rank, epoch + 1, epochs);
            fflush(stdout);
        }
    }

    // Cuối cùng, rank 1 gửi trọng số cuối cùng về rank 0 để lưu hoặc test
	/*
    if (rank == 1) {
        double* final_weights = network_get_weights(net, &weight_count);
        MPI_Send(final_weights, weight_count, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
		printf("[Rank %d] Send net to rank 0\n", rank);
        fflush(stdout);
        free(final_weights);
    } else if (rank == 0) {
        double* received_weights = (double*)malloc(sizeof(double) * weight_count);
        MPI_Recv(received_weights, weight_count, MPI_DOUBLE, 1, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		printf("[Rank %d] received net from rank 1\n", rank);
        fflush(stdout);
        network_set_weights(net, received_weights, weight_count);
        free(received_weights);
    }
	*/
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