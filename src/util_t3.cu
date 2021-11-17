#include <algorithm>
#include <helper_cuda.h>
#include <thrust/extrema.h>
#include <thrust/complex.h>
#include <thrust/execution_policy.h>

#include <cuComplex.h>
#include "memtransfer.h"
#include "../contrib/utils.h"
#include "../contrib/common.h"

void arraywidcen_gpu(BIGINT n, FLT* d_a, FLT *w, FLT *c)
// Writes out w = half-width and c = center of an interval enclosing all d_a[n]'s
// Only chooses d_a nonzero center if this increases w by less than fraction
// ARRAYWIDCEN_GROWFRAC defined in defs.h.
// This prevents rephasings which don't grow nf by much.
// If n==0, w and c are not finite.
// d_a must be readable on GPU.
{
	FLT lo, hi;

	const FLT growfac = 0.1;

	auto minMax = thrust::minmax_element(thrust::device, d_a, d_a + n);
	checkCudaErrors(cudaMemcpy(&lo,minMax.first,sizeof(FLT),cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(&hi,minMax.second,sizeof(FLT),cudaMemcpyDeviceToHost));

	*w = (hi-lo)/2;
	*c = (hi+lo)/2;
	if (std::abs(*c)<growfac*(*w)) {
		*w += std::abs(*c);
		*c = 0.0;
	}
}



void set_nhg_type3(FLT S, FLT X, cufinufft_opts opts, SPREAD_OPTS spopts,
		     BIGINT *nf, FLT *h, FLT *gam)
/* sets nf, h (upsampled grid spacing), and gamma (x_j rescaling factor),
   for type 3 only.
   Inputs:
   X and S are the xj and sk interval half-widths respectively.
   opts and spopts are the NUFFT and spreader opts strucs, respectively.
   Outputs:
   nf is the size of upsampled grid for a given single dimension.
   h is the grid spacing = 2pi/nf
   gam is the x rescale factor, ie x'_j = x_j/gam  (modulo shifts).
   Barnett 2/13/17. Caught inf/nan 3/14/17. io int types changed 3/28/17
   New logic 6/12/17
*/
{
  int nss = spopts.nspread + 1;      // since ns may be odd
  FLT Xsafe=X, Ssafe=S;              // may be tweaked locally
  if (X==0.0)                        // logic ensures XS>=1, handle X=0 a/o S=0
    if (S==0.0) {
      Xsafe=1.0;
      Ssafe=1.0;
    } else Xsafe = std::max<FLT>(Xsafe, 1/S);
  else
    Ssafe = std::max<FLT>(Ssafe, 1/X);
  // use the safe X and S...
  FLT nfd = 2.0*opts.upsampfac*Ssafe*Xsafe/PI + nss;
  if (!isfinite(nfd)) nfd=0.0;                // use FLT to catch inf
  *nf = (BIGINT)nfd;
  //printf("initial nf=%lld, ns=%d\n",*nf,spopts.nspread);
  // catch too small nf, and nan or +-inf, otherwise spread fails...
  if (*nf<2*spopts.nspread) *nf=2*spopts.nspread;
  if (*nf<MAX_NF)                             // otherwise will fail anyway
    *nf = next235even(*nf);                   // expensive at huge nf
  *h = 2*PI / *nf;                            // upsampled grid spacing
  *gam = (FLT)*nf / (2.0*opts.upsampfac*Ssafe);  // x scale fac to x'
}
