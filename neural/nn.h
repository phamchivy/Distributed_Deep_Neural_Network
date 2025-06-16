#pragma once

#include "../matrix/matrix.h"
#include "../util/img.h"
#include <stdbool.h>


// Existing NeuralNetwork structure - ADD these fields
typedef struct {
    int input;
    int hidden; 
    int output;
    double learning_rate;
    Matrix* hidden_weights;
    Matrix* output_weights;
    
    // NEW: Elastic Averaging parameters
    double alpha;           // α - elastic averaging coefficient
    double beta;            // β - server learning rate  
    bool easgd_enabled;     // Enable/disable EASGD
} NeuralNetwork;

// NEW: Elastic Center structure (master-side only)
typedef struct {
    Matrix* center_hidden_weights;   // w̄ hidden
    Matrix* center_output_weights;   // w̄ output
    bool initialized;
    int update_count;
} ElasticCenter;

NeuralNetwork* network_create(int input, int hidden, int output, double lr);
double network_train(NeuralNetwork* net, Matrix* input, Matrix* output);
// Thêm tham số epochs vào khai báo hàm
void network_train_batch_imgs(NeuralNetwork* net, Img** imgs, int batch_size, int epochs);
Matrix* network_predict_img(NeuralNetwork* net, Img* img);
double network_predict_imgs(NeuralNetwork* net, Img** imgs, int n);
Matrix* network_predict(NeuralNetwork* net, Matrix* input_data);
void network_save(NeuralNetwork* net, char* file_string);
NeuralNetwork* network_load(char* file_string);
void network_print(NeuralNetwork* net);
void network_free(NeuralNetwork* net);

// NEW: Function declarations
void network_easgd_init(NeuralNetwork* net, double alpha, double beta);
void network_apply_elastic_averaging(NeuralNetwork* net, double* center_weights, int weight_count);
void elastic_center_init(NeuralNetwork* net);
void elastic_center_update(NeuralNetwork* net, double* worker_weights, int weight_count);
void elastic_center_update_sequential(NeuralNetwork* net, double* master_weights, double* slaver_weights, int weight_count);
double* elastic_center_get_weights(int* count_out);
void elastic_center_cleanup(void);