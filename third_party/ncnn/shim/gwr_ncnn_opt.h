#pragma once
#ifdef __cplusplus
extern "C" {
#endif

typedef struct __ncnn_net_t* ncnn_net_t;

/** Configure Net options for FDnCNN (fp16 arithmetic off). */
void gwr_ncnn_configure_fdncnn(ncnn_net_t net, int num_threads);

#ifdef __cplusplus
}
#endif
