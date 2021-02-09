#ifndef __MULTI_GPU_POLICY_H__
#define __MULTI_GPU_POLICY_H__

#include <cuda.h>



CUresult get_current_device(int * i_dev, int * is_primary, int * is_clean);

#endif
