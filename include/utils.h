#ifndef FINUFFT_UTILS_H
#define FINUFFT_UTILS_H

#include <cufinufft_eitherprec.h>

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__inline__ __device__ double atomicAdd(double* address, double val)
{
	unsigned long long int* address_as_ull =
		(unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;

	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,
				__double_as_longlong(val +
					__longlong_as_double(assumed)));

		// Note: uses integer comparison to avoid hang in case of NaN
		// (since NaN != NaN)
	} while (assumed != old);

	return __longlong_as_double(old);
}
#endif

void arraywidcen_gpu(BIGINT n, FLT *a, FLT *w, FLT *c);

void set_nhg_type3(FLT S, FLT X, cufinufft_opts opts, SPREAD_OPTS spopts,
                   BIGINT *nf, FLT *h, FLT *gam);

#endif
