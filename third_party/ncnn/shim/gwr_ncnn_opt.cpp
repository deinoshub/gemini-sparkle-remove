#include "gwr_ncnn_opt.h"
#include "c_api.h"
#include "option.h"

extern "C" void gwr_ncnn_configure_fdncnn(ncnn_net_t net, int num_threads) {
    // ncnn_net_get_option returns &Net::opt (Option*).
    ncnn::Option* opt = reinterpret_cast<ncnn::Option*>(ncnn_net_get_option(net));
    opt->use_vulkan_compute = false;
    opt->use_fp16_packed = true;
    opt->use_fp16_storage = true;
    opt->use_fp16_arithmetic = false;
    opt->use_packing_layout = true;
    if (num_threads > 0) {
        opt->num_threads = num_threads;
    }
}
