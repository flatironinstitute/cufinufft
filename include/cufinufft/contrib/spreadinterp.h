#if (!defined(SPREADINTERP_H) && !defined(CUFINUFFT_SINGLE)) || \
  (!defined(SPREADINTERPF_H) && defined(CUFINUFFT_SINGLE))

#include <cmath>
#include "dataTypes.h"

#define MAX_NSPREAD 16     // upper bound on w, ie nspread, even when padded
                           // (see evaluate_kernel_vector); also for common

#undef SPREAD_OPTS

#ifdef CUFINUFFT_SINGLE
#define SPREAD_OPTS spread_optsf
#define SPREADINTERPF_H
#else
#define SPREAD_OPTS spread_opts
#define SPREADINTERP_H
#endif

struct SPREAD_OPTS {      // see cnufftspread:setup_spreader for defaults.
  int nspread;            // w, the kernel width in grid pts
  int spread_direction;   // 1 means spread NU->U, 2 means interpolate U->NU
  int pirange;            // 0: coords in [0,N), 1 coords in [-pi,pi)
  CUFINUFFT_FLT upsampfac;          // sigma, upsampling factor, default 2.0
  // ES kernel specific...
  CUFINUFFT_FLT ES_beta;
  CUFINUFFT_FLT ES_halfwidth;
  CUFINUFFT_FLT ES_c;
};

// NU coord handling macro: if p is true, rescales from [-pi,pi] to [0,N], then
// folds *only* one period below and above, ie [-N,2N], into the domain [0,N]...
#define RESCALE(x,N,p) (p ? \
		     ((x*M_1_2PI + (x<-M_PI ? 1.5 : (x>=M_PI ? -0.5 : 0.5)))*N) : \
		     (x<0 ? x+N : (x>=N ? x-N : x)))
// yuk! But this is *so* much faster than slow std::fmod that we stick to it.
CUFINUFFT_FLT evaluate_kernel(CUFINUFFT_FLT x, const SPREAD_OPTS &opts);
int setup_spreader(SPREAD_OPTS &opts, CUFINUFFT_FLT eps, CUFINUFFT_FLT upsampfac, int kerevalmeth);

#endif  // SPREADINTERP_H