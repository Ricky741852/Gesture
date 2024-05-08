// gesture_recognition_on_gemmini.c
// Created by song on 2023/04/25

#include <stdio.h>
#include <stdbool.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/func.h"
// #include "include/gesture_top_hfile.h"
#include "include/gesture_input/quantized_input_515.h"
#include "include/quantized_model_20230920.h"


int main(){
    /*****Conv1d Gesture Quantized Inference*****/
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
        perror("mlockall failed");
        exit(1);
    }
#endif
    uint64_t start, end;
    gemmini_flush(0);

    printf("\nThe correct answer of gesture class is %d\n\n", GESTURE_ANSWER);
    printf("CPU conv is turned off now\n");
    printf("Start Gemmini conv...\n\n");

    // conv1
    printf("Gemmini conv1d layer 1...\n");
    start = read_cycles();
    elem_t QConv_BN1_out[BATCH_SIZE][QConv_BN1_params.output_width][QConv_BN1_params.out_channels];
    conv1d2matmul_gemmini(BATCH_SIZE,QConv_BN1_params.input_width,QConv_BN1_params.in_channels,gesture_signals,
                          QConv_BN1_params.kernel_size, QConv_BN1_params.out_channels, QConv_BN1_params.stride_size, QConv_BN1,
                          QConv_BN1_params.output_width,QConv_BN_bias1,QConv_BN1_params.s1,(elem_t)QConv_BN1_params.z1,
                          QConv_BN1_params.s2,(elem_t)QConv_BN1_params.z2,QConv_BN1_params.sb,(elem_t)QConv_BN1_params.zb,
                          QConv_BN1_params.s4,(elem_t)QConv_BN1_params.z4,QConv_BN1_out);
    end = read_cycles();
    printf("Gemmini conv1d took %llu cycles\n", end - start);
    // block_print(BATCH_SIZE,QConv_BN1_params.output_width, QConv_BN1_params.out_channels,QConv_BN1_out);

    // conv2
    printf("Gemmini conv1d layer 2...\n");
    start = read_cycles();
    elem_t QConv_BN2_out[BATCH_SIZE][QConv_BN2_params.output_width][QConv_BN2_params.out_channels];
    conv1d2matmul_gemmini(BATCH_SIZE,QConv_BN2_params.input_width,QConv_BN2_params.in_channels,QConv_BN1_out,
                          QConv_BN2_params.kernel_size, QConv_BN2_params.out_channels, QConv_BN2_params.stride_size, QConv_BN2,
                          QConv_BN2_params.output_width,QConv_BN_bias2,QConv_BN2_params.s1,(elem_t)QConv_BN2_params.z1,
                          QConv_BN2_params.s2,(elem_t)QConv_BN2_params.z2,QConv_BN2_params.sb,(elem_t)QConv_BN2_params.zb,
                          QConv_BN2_params.s4,(elem_t)QConv_BN2_params.z4,QConv_BN2_out);
    end = read_cycles();
    printf("Gemmini conv1d took %llu cycles\n", end - start);
    // block_print(BATCH_SIZE,QConv_BN2_params.output_width, QConv_BN2_params.out_channels,QConv_BN2_out);

    // conv3
    printf("Gemmini conv1d layer 3...\n");
    start = read_cycles();
    elem_t QConv_BN3_out[BATCH_SIZE][QConv_BN3_params.output_width][QConv_BN3_params.out_channels];
    conv1d2matmul_gemmini(BATCH_SIZE,QConv_BN3_params.input_width,QConv_BN3_params.in_channels,QConv_BN2_out,
                          QConv_BN3_params.kernel_size, QConv_BN3_params.out_channels, QConv_BN3_params.stride_size, QConv_BN3,
                          QConv_BN3_params.output_width,QConv_BN_bias3,QConv_BN3_params.s1,(elem_t)QConv_BN3_params.z1,
                          QConv_BN3_params.s2,(elem_t)QConv_BN3_params.z2,QConv_BN3_params.sb,(elem_t)QConv_BN3_params.zb,
                          QConv_BN3_params.s4,(elem_t)QConv_BN3_params.z4,QConv_BN3_out);
    end = read_cycles();
    printf("Gemmini conv1d took %llu cycles\n", end - start);
    // block_print(BATCH_SIZE,QConv_BN3_params.output_width, QConv_BN3_params.out_channels,QConv_BN3_out);

    // conv4
    printf("Gemmini conv1d layer 4...\n");
    start = read_cycles();
    elem_t QConv_BN4_out[BATCH_SIZE][QConv_BN4_params.output_width][QConv_BN4_params.out_channels];
    conv1d2matmul_gemmini(BATCH_SIZE,QConv_BN4_params.input_width,QConv_BN4_params.in_channels,QConv_BN3_out,
                          QConv_BN4_params.kernel_size, QConv_BN4_params.out_channels, QConv_BN4_params.stride_size, QConv_BN4,
                          QConv_BN4_params.output_width,QConv_BN_bias4,QConv_BN4_params.s1,(elem_t)QConv_BN4_params.z1,
                          QConv_BN4_params.s2,(elem_t)QConv_BN4_params.z2,QConv_BN4_params.sb,(elem_t)QConv_BN4_params.zb,
                          QConv_BN4_params.s4,(elem_t)QConv_BN4_params.z4,QConv_BN4_out);
    end = read_cycles();
    printf("Gemmini conv1d took %llu cycles\n\n", end - start);
    // block_print(BATCH_SIZE,QConv_BN4_params.output_width, QConv_BN4_params.out_channels,QConv_BN4_out);
    int ans = find_max_channel(BATCH_SIZE,QConv_BN4_params.output_width, QConv_BN4_params.out_channels,QConv_BN4_out);
    printf("\n");
    printf("Gesture Answer, Predict Answer\n");
    printf("%d,%d", GESTURE_ANSWER, ans);
    printf("\n");


}
