#pragma once
#ifdef __cplusplus
extern "C" {
#endif

typedef struct __ncnn_net_t* ncnn_net_t;

/** Configure Net options to match GWT/Python FDnCNN (fp16 arith off). */
void gwr_ncnn_configure_fdncnn(ncnn_net_t net, int num_threads);

#ifdef __cplusplus
}
#endif
