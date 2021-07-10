#ifndef __MULTI_GPU_POLICY_H__
#define __MULTI_GPU_POLICY_H__

#include <cuda.h>
#include <cufinufft_opts.h>



typedef struct CtxProfile {
    int i_dev;
    int is_primary;
    int is_clean;
} CtxProfile;



CUresult get_current_device(CtxProfile * ctx_profile);
int use_set_device(CtxProfile * ctx_profile, cufinufft_opts * opts);
int policy_set_device(cufinufft_opts * opts);

#endif
