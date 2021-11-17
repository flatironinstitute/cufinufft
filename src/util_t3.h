#ifndef __UTIL_T3_H__
#define __UTIL_T3_H__

#include <cufinufft_eitherprec.h>


void arraywidcen_gpu(BIGINT n, FLT* a, FLT *w, FLT *c);

void set_nhg_type3(FLT S, FLT X, cufinufft_opts opts, SPREAD_OPTS spopts,
		     BIGINT *nf, FLT *h, FLT *gam);


#endif
