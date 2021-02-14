#include <multi_gpu_policy.h>
#include <cufinufft_opts.h>
#include <cuda.h>



CUresult get_current_device(CtxProfile * ctx_profile) {

    cudaError_t cuda_err;
    CUresult ierr;

    // GET current device bound to this thread (this will work for _both_ the
    // cuda runtim API and the cuda driver)
    cuda_err = cudaGetDevice(& ctx_profile->i_dev);
    if (cuda_err != cudaSuccess) {
        ctx_profile->i_dev      = -1;
        ctx_profile->is_primary = -1;
        ctx_profile->is_clean   = -1;
        return CUDA_ERROR_INVALID_CONTEXT;
    }

    // GET the state of the primary context
    unsigned int flags;
    int active;
    cuDevicePrimaryCtxGetState(ctx_profile->i_dev, & flags, & active);
    if (active == 1){
        ctx_profile->is_primary = 1;
        ctx_profile->is_clean   = 0;
        return CUDA_SUCCESS;
    }

    // The PRIMARY CONTEXT could be the only context on the device bound to
    // this thread -- and just not be active because nothing has been called
    // it, or another context is the current context

    CUdevice device;

    ierr = cuCtxGetDevice(& device);
    if (ierr == CUDA_ERROR_INVALID_CONTEXT) {
        ctx_profile->is_primary = 1;
        ctx_profile->is_clean   = 1;
    } else if (ierr != CUDA_SUCCESS) {
        return ierr;
    }

    // There is defintely a device bound to this thread -- so there MUST be a
    // current context. The only thing to figure out now is if this context is
    // the primary context (which just hasn't been used yet), or another.

    CUcontext context;

    ierr = cuCtxGetCurrent(& context);
    if (ierr != CUDA_SUCCESS)
        return ierr;

    CUcontext primary_context;
    // NOTE: this will make the primary context active -- we need to release it
    // again below:
    ierr = cuDevicePrimaryCtxRetain(& primary_context, device);
    if (ierr != CUDA_SUCCESS)
        return ierr;
    // Restore the pre-existing context by de-activating the primary context
    // that cudaPrimaryCtxRetain activated
    ierr = cuDevicePrimaryCtxRelease(device);
    if (ierr != CUDA_SUCCESS)
        return ierr;

    if (primary_context == context) {
        ctx_profile->is_primary = 1;
        ctx_profile->is_clean   = 0;
        return CUDA_SUCCESS;
    }

    ctx_profile->is_primary = 0;
    ctx_profile->is_clean   = 0;
    return CUDA_SUCCESS;
}



int use_set_device(CtxProfile * ctx_profile, cufinufft_opts * opts) {

    if (ctx_profile->is_primary == 1 || ctx_profile->is_clean == 1 || opts->gpu_force_primary_ctx == 1) {
        return 1;
    }

    return 0;
}
