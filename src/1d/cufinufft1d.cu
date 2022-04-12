#include <iostream>
#include <iomanip>
#include <math.h>
#include "cuda_hip_wrapper.h"
#include <helper_cuda.h>
#include <complex>

#include <cufinufft_eitherprec.h>
#include "../cuspreadinterp.h"
#include "../cudeconvolve.h"
#include "../memtransfer.h"

using namespace std;

int CUFINUFFT1D1_EXEC(CUCPX* d_c, CUCPX* d_fk, CUFINUFFT_PLAN d_plan)
/*  
	1D Type-1 NUFFT

	This function is called in "exec" stage (See ../cufinufft.cu).
	It includes (copied from doc in finufft library)
		Step 1: spread data to oversampled regular mesh using kernel
		Step 2: compute FFT on uniform mesh
		Step 3: deconvolve by division of each Fourier mode independently by the
		        Fourier series coefficient of the kernel.

	Melody Shih 11/21/21
*/
{
	assert(d_plan->spopts.spread_direction == 1);
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	int blksize;
	int ier;
	CUCPX* d_fkstart;
	CUCPX* d_cstart;
	for(int i=0; i*d_plan->maxbatchsize < d_plan->ntransf; i++){
		blksize = min(d_plan->ntransf - i*d_plan->maxbatchsize, 
			d_plan->maxbatchsize);
		d_cstart   = d_c + i*d_plan->maxbatchsize*d_plan->M;
		d_fkstart  = d_fk + i*d_plan->maxbatchsize*d_plan->ms;
		d_plan->c  = d_cstart;
		d_plan->fk = d_fkstart;

		checkCudaErrors(cudaMemset(d_plan->fw,0,d_plan->maxbatchsize*
					d_plan->nf1*sizeof(CUCPX)));// this is needed
#ifdef TIME
		float milliseconds = 0;
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("[time  ] \tInitialize fw to 0\t %.3g s\n", 
			milliseconds/1000);
#endif
		// Step 1: Spread
		cudaEventRecord(start);
		ier = CUSPREAD1D(d_plan,blksize);
		if(ier != 0 ){
			printf("error: cuspread1d, method(%d)\n", d_plan->opts.gpu_method);
			return ier;
		}
#ifdef TIME
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("[time  ] \tSpread (%d)\t\t %.3g s\n", milliseconds/1000, 
			d_plan->opts.gpu_method);
#endif
		// Step 2: FFT
		cudaEventRecord(start);
		CUFFT_EX(d_plan->fftplan, d_plan->fw, d_plan->fw, d_plan->iflag);
#ifdef TIME
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("[time  ] \tCUFFT Exec\t\t %.3g s\n", milliseconds/1000);
#endif

		// Step 3: deconvolve and shuffle
		cudaEventRecord(start);
		CUDECONVOLVE1D(d_plan,blksize);
#ifdef TIME
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("[time  ] \tDeconvolve\t\t %.3g s\n", milliseconds/1000);
#endif
	}
	return ier;
}

int CUFINUFFT1D2_EXEC(CUCPX* d_c, CUCPX* d_fk, CUFINUFFT_PLAN d_plan)
/*  
	1D Type-2 NUFFT

	This function is called in "exec" stage (See ../cufinufft.cu).
	It includes (copied from doc in finufft library)
		Step 1: deconvolve (amplify) each Fourier mode, dividing by kernel 
		        Fourier coeff
		Step 2: compute FFT on uniform mesh
		Step 3: interpolate data to regular mesh

	Melody Shih 11/21/21
*/
{
	assert(d_plan->spopts.spread_direction == 2);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	int blksize;
	int ier;
	CUCPX* d_fkstart;
	CUCPX* d_cstart;
	for(int i=0; i*d_plan->maxbatchsize < d_plan->ntransf; i++){
		blksize = min(d_plan->ntransf - i*d_plan->maxbatchsize, 
			d_plan->maxbatchsize);
		d_cstart  = d_c  + i*d_plan->maxbatchsize*d_plan->M;
		d_fkstart = d_fk + i*d_plan->maxbatchsize*d_plan->ms;

		d_plan->c = d_cstart;
		d_plan->fk = d_fkstart;

		// Step 1: amplify Fourier coeffs fk and copy into upsampled array fw
		cudaEventRecord(start);
		CUDECONVOLVE1D(d_plan,blksize);
#ifdef TIME
		float milliseconds = 0;
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("[time  ] \tAmplify & Copy fktofw\t %.3g s\n", milliseconds/1000);
#endif
		// Step 2: FFT
		cudaDeviceSynchronize();
		cudaEventRecord(start);
		CUFFT_EX(d_plan->fftplan, d_plan->fw, d_plan->fw, d_plan->iflag);
#ifdef TIME
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("[time  ] \tCUFFT Exec\t\t %.3g s\n", milliseconds/1000);
#endif

		// Step 3: deconvolve and shuffle
		cudaEventRecord(start);
		ier = CUINTERP1D(d_plan, blksize);
		if(ier != 0 ){
			printf("error: cuinterp1d, method(%d)\n", d_plan->opts.gpu_method);
			return ier;
		}
#ifdef TIME
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("[time  ] \tUnspread (%d)\t\t %.3g s\n", milliseconds/1000,
			d_plan->opts.gpu_method);
#endif
	}
	return ier;
}

