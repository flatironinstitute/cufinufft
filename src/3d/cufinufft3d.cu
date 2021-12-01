#include <iostream>
#include <iomanip>
#include <math.h>
#include <helper_cuda.h>
#include <complex>
#include <cufft.h>
#include <thrust/transform.h>
#include <thrust/complex.h>
#include <thrust/execution_policy.h>

#include <cufinufft_eitherprec.h>
#include "../cuspreadinterp.h"
#include "../cudeconvolve.h"
#include "../memtransfer.h"

using namespace std;

int CUFINUFFT3D1_EXEC(CUCPX* d_c, CUCPX* d_fk, CUFINUFFT_PLAN d_plan)
/*  
	3D Type-1 NUFFT

	This function is called in "exec" stage (See ../cufinufft.cu).
	It includes (copied from doc in finufft library)
		Step 1: spread data to oversampled regular mesh using kernel
		Step 2: compute FFT on uniform mesh
		Step 3: deconvolve by division of each Fourier mode independently by the
		        Fourier series coefficient of the kernel.

	Melody Shih 07/25/19		
*/
{
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
		d_cstart = d_c + i*d_plan->maxbatchsize*d_plan->M;
		d_fkstart = d_fk + i*d_plan->maxbatchsize*d_plan->ms*d_plan->mt*
			d_plan->mu;

		d_plan->c = d_cstart;
		d_plan->fk = d_fkstart;

		checkCudaErrors(cudaMemset(d_plan->fw,0,d_plan->maxbatchsize*
					d_plan->nf1*d_plan->nf2*d_plan->nf3*sizeof(CUCPX)));
#ifdef TIME
		float milliseconds = 0;
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("[time  ] \tInitialize fw\t\t %.3g s\n", milliseconds/1000);
#endif
		// Step 1: Spread
		cudaEventRecord(start);
	    ier = CUSPREAD3D(d_plan, blksize, d_plan->c, d_plan->fw);
		if(ier != 0 ){
			printf("error: cuspread3d, method(%d)\n", d_plan->opts.gpu_method);
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
		CUDECONVOLVE3D(d_plan, blksize);
#ifdef TIME
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("[time  ] \tDeconvolve\t\t %.3g s\n", milliseconds/1000);
#endif
	}
	return ier;
}

int CUFINUFFT3D2_EXEC(CUCPX* d_c, CUCPX* d_fk, CUFINUFFT_PLAN d_plan)
/*  
	3D Type-2 NUFFT

	This function is called in "exec" stage (See ../cufinufft.cu).
	It includes (copied from doc in finufft library)
		Step 1: deconvolve (amplify) each Fourier mode, dividing by kernel 
		        Fourier coeff
		Step 2: compute FFT on uniform mesh
		Step 3: interpolate data to regular mesh

	Melody Shih 07/25/19		
*/
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int blksize;
	int ier;
	CUCPX* d_fkstart;
	CUCPX* d_cstart;
	for(int i=0; i*d_plan->maxbatchsize < d_plan->ntransf; i++){
		blksize = min(d_plan->ntransf - i*d_plan->maxbatchsize, 
			d_plan->maxbatchsize);
		d_cstart  = d_c  + i*d_plan->maxbatchsize*d_plan->M;
		d_fkstart = d_fk + i*d_plan->maxbatchsize*d_plan->ms*d_plan->mt*
			d_plan->mu;

		d_plan->c = d_cstart;
		d_plan->fk = d_fkstart;

		// Step 1: amplify Fourier coeffs fk and copy into upsampled array fw
		cudaEventRecord(start);
		CUDECONVOLVE3D(d_plan, blksize);
#ifdef TIME
		float milliseconds = 0;
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("[time  ] \tAmplify & Copy fktofw\t %.3g s\n", milliseconds/1000);
#endif
		// Step 2: FFT
		cudaEventRecord(start);
		cudaDeviceSynchronize();
		CUFFT_EX(d_plan->fftplan, d_plan->fw, d_plan->fw, d_plan->iflag);
#ifdef TIME
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("[time  ] \tCUFFT Exec\t\t %.3g s\n", milliseconds/1000);
#endif

		// Step 3: deconvolve and shuffle
		cudaEventRecord(start);
		ier = CUINTERP3D(d_plan, blksize);
		if(ier != 0 ){
			printf("error: cuinterp3d, method(%d)\n", d_plan->opts.gpu_method);
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

int CUFINUFFT3D3_EXEC(CUCPX* d_c, CUCPX* d_fk, CUFINUFFT_PLAN d_plan)
/*
	3D Type-3 NUFFT

*/
{
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
		d_cstart = d_c + i*d_plan->maxbatchsize*d_plan->M;
		d_fkstart = d_fk + i*d_plan->maxbatchsize*d_plan->N;

		d_plan->c = d_cstart;
		d_plan->fk = d_fkstart;

		cudaEventRecord(start);

		// Step 0: prephase
		if (d_plan->t3P.D1!=0.0 || d_plan->t3P.D2!=0.0 || d_plan->t3P.D3!=0.0) {
			for (int i=0; i<blksize; i++) {
				thrust::transform(thrust::device, reinterpret_cast<const thrust::complex<FLT>*>(d_plan->prephase),
						reinterpret_cast<const thrust::complex<FLT>*>(d_plan->prephase) + d_plan->M,
						reinterpret_cast<const thrust::complex<FLT>*>(d_cstart + i * d_plan->M),
						reinterpret_cast<thrust::complex<FLT>*>(d_plan->cpbatch + i * d_plan->M),
						thrust::multiplies<thrust::complex<FLT>>());
			}
		}

		checkCudaErrors(cudaMemset(d_plan->fw,0,d_plan->maxbatchsize*
					d_plan->nf1*d_plan->nf2*d_plan->nf3*sizeof(CUCPX)));
#ifdef TIME
		float milliseconds = 0;
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("[time  ] \tInitialize fw\t\t %.3g s\n", milliseconds/1000);
#endif
		// Step 1: Spread
		cudaEventRecord(start);
		CUCPX* sp_input = d_cstart;
		if (d_plan->t3P.D1!=0.0 || d_plan->t3P.D2!=0.0 || d_plan->t3P.D3!=0.0) {
			sp_input = d_plan->cpbatch;
		}
	    ier = CUSPREAD3D(d_plan, blksize, sp_input, d_plan->fw);

		if(ier != 0 ){
			printf("error: cuspread3d, method(%d)\n", d_plan->opts.gpu_method);
			return ier;
		}
#ifdef TIME
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("[time  ] \tSpread (%d)\t\t %.3g s\n", milliseconds/1000, 
			d_plan->opts.gpu_method);
#endif

		// Step 2: Execute type 2
		d_plan->innert2plan->ntransf = blksize;

		cudaEventRecord(start);
		CUFINUFFT3D2_EXEC(d_fkstart, d_plan->fw, d_plan->innert2plan);
#ifdef TIME
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("[time  ] \tInner type 2 execution (%d)\t\t %.3g s\n", milliseconds/1000, 
			d_plan->opts.gpu_method);
#endif

		// Step 3: deconvolve
		cudaEventRecord(start);

		for (int i=0; i<blksize; i++) {
			thrust::transform(thrust::device, reinterpret_cast<const thrust::complex<FLT>*>(d_plan->deconv),
					reinterpret_cast<const thrust::complex<FLT>*>(d_plan->deconv) + d_plan->N,
					reinterpret_cast<thrust::complex<FLT>*>(d_fkstart + i * d_plan->N),
					reinterpret_cast<thrust::complex<FLT>*>(d_fkstart + i * d_plan->N),
					thrust::multiplies<thrust::complex<FLT>>());
		}
#ifdef TIME
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("[time  ] \tDeconvolve\t\t %.3g s\n", milliseconds/1000);
#endif

	}

	return ier;
}


