#include <iostream>
#include <iomanip>
#include <vector>
#include <complex>
#include <cstddef>
#include <math.h>
#include <helper_cuda.h>
#include <complex>
#include <cufft.h>
// #include <thrust/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/complex.h>

#include <cufinufft_eitherprec.h>
#include "cuspreadinterp.h"
#include "cudeconvolve.h"
#include "memtransfer.h"
#include "util_t3.h"

using namespace std;

template <typename T>
struct t3_rescale {
	__host__ __device__ T operator()(const T& x) const {
            return (x - C) * ig;
        }

	T C;
	T ig;
};

template <typename T>
struct t3_prephase_d1 {
	__host__ __device__ thrust::complex<T> operator()(const thrust::tuple<T>& x) const {
            return thrust::exp(thrust::complex<T>(0, sign * D1 * thrust::get<0>(x)));
        }

	T sign;
	T D1;

};

template <typename T>
struct t3_prephase_d2 {
	__host__ __device__ thrust::complex<T> operator()(const thrust::tuple<T, T>& x) const {
            return thrust::exp(thrust::complex<T>(0, sign * (D1 * thrust::get<0>(x) + D2 * thrust::get<1>(x))));
        }

	T sign;
	T D1;
	T D2;

};

template <typename T>
struct t3_prephase_d3 {
	__host__ __device__ thrust::complex<T> operator()(const thrust::tuple<T, T, T>& x) const {
            return thrust::exp(thrust::complex<T>(0, sign * (D1 * thrust::get<0>(x) + D2 * thrust::get<1>(x) + 
					    D3 * thrust::get<2>(x))));
        }

	T sign;
	T D1;
	T D2;
	T D3;
};

void SETUP_BINSIZE(int type, int dim, cufinufft_opts *opts)
{
	switch(dim)
	{
		case 1:
		{
			opts->gpu_binsizex = (opts->gpu_binsizex < 0) ? 1024:
				opts->gpu_binsizex;
			opts->gpu_binsizey = 1;
			opts->gpu_binsizez = 1;
		}
		break;
		case 2:
		{
			opts->gpu_binsizex = (opts->gpu_binsizex < 0) ? 32:
				opts->gpu_binsizex;
			opts->gpu_binsizey = (opts->gpu_binsizey < 0) ? 32:
				opts->gpu_binsizey;
			opts->gpu_binsizez = 1;
		}
		break;
		case 3:
		{
			switch(opts->gpu_method)
			{
				case 1:
				case 2:
				{
					opts->gpu_binsizex = (opts->gpu_binsizex < 0) ? 16:
						opts->gpu_binsizex;
					opts->gpu_binsizey = (opts->gpu_binsizey < 0) ? 16:
						opts->gpu_binsizey;
					opts->gpu_binsizez = (opts->gpu_binsizez < 0) ? 2:
						opts->gpu_binsizez;
				}
				break;
				case 4:
				{
					opts->gpu_obinsizex = (opts->gpu_obinsizex < 0) ? 8:
						opts->gpu_obinsizex;
					opts->gpu_obinsizey = (opts->gpu_obinsizey < 0) ? 8:
						opts->gpu_obinsizey;
					opts->gpu_obinsizez = (opts->gpu_obinsizez < 0) ? 8:
						opts->gpu_obinsizez;
					opts->gpu_binsizex = (opts->gpu_binsizex < 0) ? 4:
						opts->gpu_binsizex;
					opts->gpu_binsizey = (opts->gpu_binsizey < 0) ? 4:
						opts->gpu_binsizey;
					opts->gpu_binsizez = (opts->gpu_binsizez < 0) ? 4:
						opts->gpu_binsizez;
				}
				break;
			}
		}
		break;
	}
}

#ifdef __cplusplus
extern "C" {
#endif
int CUFINUFFT_MAKEPLAN(int type, int dim, int *nmodes, int iflag,
		       int ntransf, FLT tol, int maxbatchsize,
		       CUFINUFFT_PLAN *d_plan_ptr, cufinufft_opts *opts)
/*
	"plan" stage (in single or double precision).
        See ../docs/cppdoc.md for main user-facing documentation.
        Note that *d_plan_ptr in the args list was called simply *plan there.
        This is the remaining dev-facing doc:

This performs:
		(0) creating a new plan struct (d_plan), a pointer to which is passed
		    back by writing that pointer into *d_plan_ptr.
		(1) set up the spread option, d_plan.spopts.
		(2) calculate the correction factor on cpu, copy the value from cpu to
		    gpu
		(3) allocate gpu arrays with size determined by number of fourier modes
		    and method related options that had been set in d_plan.opts
		(4) call cufftPlanMany and save the cufft plan inside cufinufft plan
        Variables and arrays inside the plan struct are set and allocated.

	Melody Shih 07/25/19. Use-facing moved to markdown, Barnett 2/16/21.
*/
{
	int ier = 0;

	/* allocate the plan structure, assign address to user pointer. */
	CUFINUFFT_PLAN d_plan = new CUFINUFFT_PLAN_S;
	*d_plan_ptr = d_plan;
        // Zero out your struct, (sets all pointers to NULL)
	memset(d_plan, 0, sizeof(*d_plan));

	int fftsign = (iflag>=0) ? 1 : -1;

	d_plan->dim = dim;
	d_plan->iflag = fftsign;
	d_plan->ntransf = ntransf;
	if (maxbatchsize==0)                    // implies: use a heuristic.
	   maxbatchsize = min(ntransf, 8);      // heuristic from test codes
	d_plan->maxbatchsize = maxbatchsize;
	d_plan->type = type;
	d_plan->tol = tol;


	/* If a user has not supplied their own options, assign defaults for them. */
	if (opts==NULL){    // use default opts
	  ier = CUFINUFFT_DEFAULT_OPTS(type, dim, &(d_plan->opts));
	  if (ier != 0){
	    printf("error: CUFINUFFT_DEFAULT_OPTS returned error %d.\n", ier);
	    return ier;
	  }
	} else {    // or read from what's passed in
	  d_plan->opts = *opts;    // keep a deep copy; changing *opts now has no effect
	}

	/* Setup Spreader */
	ier = setup_spreader_for_nufft(d_plan->spopts,tol,d_plan->opts);
	if (ier>1)                           // proceed if success or warning
	  return ier;


	// Setup for type 3 is done when points are set. Only implemented for dim = 3.
	if(d_plan->type == 3 && d_plan->dim == 3) 
		return ier;

        // Mult-GPU support: set the CUDA Device ID:
        int orig_gpu_device_id;
        cudaGetDevice(& orig_gpu_device_id);
        if (opts == NULL) {
            // options might not be supplied to this function => assume device
            // 0 by default
            cudaSetDevice(0);
        } else {
            cudaSetDevice(opts->gpu_device_id);
        }

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	d_plan->ms = nmodes[0];
	d_plan->mt = nmodes[1];
	d_plan->mu = nmodes[2];

	SETUP_BINSIZE(type, dim, &d_plan->opts);
	int nf1=1, nf2=1, nf3=1;
	SET_NF_TYPE12(d_plan->ms, d_plan->opts, d_plan->spopts, &nf1,
				  d_plan->opts.gpu_obinsizex);
	if(dim > 1)
		SET_NF_TYPE12(d_plan->mt, d_plan->opts, d_plan->spopts, &nf2,
                      d_plan->opts.gpu_obinsizey);
	if(dim > 2)
		SET_NF_TYPE12(d_plan->mu, d_plan->opts, d_plan->spopts, &nf3,
                      d_plan->opts.gpu_obinsizez);

	d_plan->nf1 = nf1;
	d_plan->nf2 = nf2;
	d_plan->nf3 = nf3;

	if(d_plan->type == 1)
		d_plan->spopts.spread_direction = 1;
	if(d_plan->type == 2)
		d_plan->spopts.spread_direction = 2;
	// this may move to gpu
	CNTime timer; timer.start();
	FLT *fwkerhalf1, *fwkerhalf2, *fwkerhalf3;

	fwkerhalf1 = (FLT*)malloc(sizeof(FLT)*(nf1/2+1));
	onedim_fseries_kernel(nf1, fwkerhalf1, d_plan->spopts);
	if(dim > 1){
		fwkerhalf2 = (FLT*)malloc(sizeof(FLT)*(nf2/2+1));
		onedim_fseries_kernel(nf2, fwkerhalf2, d_plan->spopts);
	}
	if(dim > 2){
		fwkerhalf3 = (FLT*)malloc(sizeof(FLT)*(nf3/2+1));
		onedim_fseries_kernel(nf3, fwkerhalf3, d_plan->spopts);
	}
#ifdef TIME
	printf("[time  ] \tkernel fser (ns=%d):\t %.3g s\n", d_plan->spopts.nspread,
		timer.elapsedsec());
#endif

	cudaEventRecord(start);
	switch(d_plan->dim)
	{
		case 1:
		{
			ier = ALLOCGPUMEM1D_PLAN(d_plan);
		}
		break;
		case 2:
		{
			ier = ALLOCGPUMEM2D_PLAN(d_plan);
		}
		break;
		case 3:
		{
			ier = ALLOCGPUMEM3D_PLAN(d_plan);
		}
		break;
	}
#ifdef TIME
	float milliseconds = 0;
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tAllocate GPU memory plan %.3g s\n", milliseconds/1000);
#endif

	cudaEventRecord(start);
	checkCudaErrors(cudaMemcpy(d_plan->fwkerhalf1,fwkerhalf1,(nf1/2+1)*
		sizeof(FLT),cudaMemcpyHostToDevice));
	if(dim > 1)
		checkCudaErrors(cudaMemcpy(d_plan->fwkerhalf2,fwkerhalf2,(nf2/2+1)*
			sizeof(FLT),cudaMemcpyHostToDevice));
	if(dim > 2)
		checkCudaErrors(cudaMemcpy(d_plan->fwkerhalf3,fwkerhalf3,(nf3/2+1)*
			sizeof(FLT),cudaMemcpyHostToDevice));
#ifdef TIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tCopy fwkerhalf1,2 HtoD\t %.3g s\n", milliseconds/1000);
#endif

	cudaEventRecord(start);
	cufftHandle fftplan;
	switch(d_plan->dim)
	{
		case 1:
		{
			int n[] = {nf1};
			int inembed[] = {nf1};

			cufftPlanMany(&fftplan,1,n,inembed,1,inembed[0],
				inembed,1,inembed[0],CUFFT_TYPE,maxbatchsize);
		}
		break;
		case 2:
		{
			int n[] = {nf2, nf1};
			int inembed[] = {nf2, nf1};

			cufftPlanMany(&fftplan,2,n,inembed,1,inembed[0]*inembed[1],
				inembed,1,inembed[0]*inembed[1],CUFFT_TYPE,maxbatchsize);
		}
		break;
		case 3:
		{
			int n[] = {nf3, nf2, nf1};
			int inembed[] = {nf3, nf2, nf1};

			cufftPlanMany(&fftplan,3,n,inembed,1,inembed[0]*inembed[1]*
				inembed[2],inembed,1,inembed[0]*inembed[1]*inembed[2],
				CUFFT_TYPE,maxbatchsize);
		}
		break;
	}
	d_plan->fftplan = fftplan;
#ifdef TIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tCUFFT Plan\t\t %.3g s\n", milliseconds/1000);
#endif
	free(fwkerhalf1);
	if(dim > 1)
		free(fwkerhalf2);
	if(dim > 2)
		free(fwkerhalf3);

        // Multi-GPU support: reset the device ID
        cudaSetDevice(orig_gpu_device_id);

	return ier;
}

int CUFINUFFT_SETPTS(int M, FLT* d_kx, FLT* d_ky, FLT* d_kz, int N, FLT *d_s,
	FLT *d_t, FLT *d_u, CUFINUFFT_PLAN d_plan)
/*
	"setNUpts" stage (in single or double precision).

	In this stage, we
		(1) set the number and locations of nonuniform points
		(2) allocate gpu arrays with size determined by number of nupts
		(3) rescale x,y,z coordinates for spread/interp (on gpu, rescaled
		    coordinates are stored)
		(4) determine the spread/interp properties that only relates to the
		    locations of nupts (see 2d/spread2d_wrapper.cu,
		    3d/spread3d_wrapper.cu for what have been done in
		    function spread<dim>d_<method>_prop() )

        See ../docs/cppdoc.md for main user-facing documentation.
        Here is the old developer docs, which are useful only to translate
        the argument names from the user-facing ones:
        
	Input:
	M                 number of nonuniform points
	d_kx, d_ky, d_kz  gpu array of x,y,z locations of sources (each a size M
	                  FLT array) in [-pi, pi). set h_kz to "NULL" if dimension
	                  is less than 3. same for h_ky for dimension 1.
	N, d_s, d_t, d_u  not used for type1, type2. set to 0 and NULL.

	Input/Output:
	d_plan            pointer to a CUFINUFFT_PLAN_S. Variables and arrays inside
	                  the plan are set and allocated.

        Returned value:
        a status flag: 0 if success, otherwise an error occurred

Notes: the type FLT means either single or double, matching the
	precision of the library version called.

	Melody Shih 07/25/19; Barnett 2/16/21 moved out docs.
*/
{
        // Mult-GPU support: set the CUDA Device ID:
        int orig_gpu_device_id;
        cudaGetDevice(& orig_gpu_device_id);
        cudaSetDevice(d_plan->opts.gpu_device_id);

	int ier;

	int dim = d_plan->dim;

	d_plan->M = M;

	if(d_plan->type == 3) {
		d_plan->N = N;

		// pick x, s intervals & shifts & # fine grid pts (nf) in each dim...
		FLT S1,S2,S3;       // get half-width X, center C, which contains {x_j}...
		arraywidcen_gpu(M, d_kx, &(d_plan->t3P.X1), &(d_plan->t3P.C1));
		arraywidcen_gpu(N, d_s, &S1, &(d_plan->t3P.D1));      // same D, S, but for {s_k}
		set_nhg_type3(S1, d_plan->t3P.X1, d_plan->opts, d_plan->spopts,
			&(d_plan->nf1), &(d_plan->t3P.h1), &(d_plan->t3P.gam1));  // applies twist i)

		d_plan->t3P.C2 = 0.0;        // their defaults if dim 2 unused, etc
		d_plan->t3P.D2 = 0.0;
		if (d_plan->dim>1) {
			arraywidcen_gpu(M, d_ky, &(d_plan->t3P.X2), &(d_plan->t3P.C2));     // {y_j}
			arraywidcen_gpu(N, d_t, &S2, &(d_plan->t3P.D2));               // {t_k}
			set_nhg_type3(S2, d_plan->t3P.X2, d_plan->opts, d_plan->spopts, &(d_plan->nf2),
			    &(d_plan->t3P.h2), &(d_plan->t3P.gam2));
		}

		d_plan->t3P.C3 = 0.0;
		d_plan->t3P.D3 = 0.0;
		if (d_plan->dim>2) {
			arraywidcen_gpu(M, d_kz, &(d_plan->t3P.X3), &(d_plan->t3P.C3));     // {z_j}
			arraywidcen_gpu(N, d_u, &S3, &(d_plan->t3P.D3));               // {u_k}
			set_nhg_type3(S3, d_plan->t3P.X3, d_plan->opts,d_plan->spopts,
			    &(d_plan->nf3), &(d_plan->t3P.h3), &(d_plan->t3P.gam3));
		}

		int nf1 = d_plan->nf1;
		int nf2 = d_plan->nf2;
		int nf3 = d_plan->nf3;

		// Allocate memory
		checkCudaErrors(cudaMalloc(&d_plan->cpbatch, d_plan->maxbatchsize*nf1*nf2*nf3*
			sizeof(CUCPX)));

		switch(d_plan->dim)
		{
			case 1:
			{
				ier = ALLOCGPUMEM1D_PLAN(d_plan);
			}
			break;
			case 2:
			{
				ier = ALLOCGPUMEM2D_PLAN(d_plan);
			}
			break;
			case 3:
			{
				ier = ALLOCGPUMEM3D_PLAN(d_plan);
			}
			break;
		}

		checkCudaErrors(cudaMalloc(&d_plan->kx, M * sizeof(FLT)));
		if(d_plan->dim > 1) checkCudaErrors(cudaMalloc(&d_plan->ky, M * sizeof(FLT)));
		if(d_plan->dim > 2) checkCudaErrors(cudaMalloc(&d_plan->kz, M * sizeof(FLT)));

		checkCudaErrors(cudaMalloc(&d_plan->s, N * sizeof(FLT)));
		if(d_plan->dim > 1) checkCudaErrors(cudaMalloc(&d_plan->t, N * sizeof(FLT)));
		if(d_plan->dim > 2) checkCudaErrors(cudaMalloc(&d_plan->u, N * sizeof(FLT)));


		checkCudaErrors(cudaMalloc(&d_plan->prephase, M * sizeof(CUCPX)));
		checkCudaErrors(cudaMalloc(&d_plan->deconv, N * sizeof(CUCPX)));

		// rescale
		thrust::transform(thrust::device, d_kx, d_kx + M, d_plan->kx,
				  t3_rescale<FLT>{d_plan->t3P.C1, (FLT) (1.0 / d_plan->t3P.gam1)});
		if(d_plan->dim > 1)
			thrust::transform(thrust::device, d_ky, d_ky + M, d_plan->ky,
					  t3_rescale<FLT>{d_plan->t3P.C2, (FLT) (1.0 / d_plan->t3P.gam2)});
		if(d_plan->dim > 2)
			thrust::transform(thrust::device, d_kz, d_kz + M, d_plan->kz,
					  t3_rescale<FLT>{d_plan->t3P.C3, (FLT) (1.0 / d_plan->t3P.gam3)});

		thrust::transform(thrust::device, d_s, d_s + N, d_plan->s,
				  t3_rescale<FLT>{d_plan->t3P.D1, d_plan->t3P.h1 * d_plan->t3P.gam1});
		if(d_plan->dim > 1)
			thrust::transform(thrust::device, d_t, d_t + N, d_plan->t,
					  t3_rescale<FLT>{d_plan->t3P.D2, d_plan->t3P.h2 * d_plan->t3P.gam2});
		if(d_plan->dim > 2)
			thrust::transform(thrust::device, d_u, d_u + N, d_plan->u,
					  t3_rescale<FLT>{d_plan->t3P.D3, d_plan->t3P.h3 * d_plan->t3P.gam3});

		// compute prephase
		FLT imasign = (d_plan->iflag>=0) ? 1 : -1;             // +-i
		if (d_plan->t3P.D1!=0.0 || d_plan->t3P.D2!=0.0 || d_plan->t3P.D3!=0.0) {
			if (d_plan->dim == 1) { 
				auto it = thrust::make_zip_iterator(thrust::make_tuple(d_kx));
				thrust::transform(thrust::device, it, it + M, reinterpret_cast<thrust::complex<FLT>*>(d_plan->prephase),
						  t3_prephase_d1<FLT>{imasign, d_plan->t3P.D1});
			} else if (d_plan->dim == 2) {
				auto it = thrust::make_zip_iterator(thrust::make_tuple(d_kx, d_ky));
				thrust::transform(thrust::device, it, it + M, reinterpret_cast<thrust::complex<FLT>*>(d_plan->prephase),
						  t3_prephase_d2<FLT>{imasign, d_plan->t3P.D1, d_plan->t3P.D2});
			} else {
				auto it = thrust::make_zip_iterator(thrust::make_tuple(d_kx, d_ky, d_kz));
				thrust::transform(thrust::device, it, it + M, reinterpret_cast<thrust::complex<FLT>*>(d_plan->prephase),
						  t3_prephase_d3<FLT>{imasign, d_plan->t3P.D1, d_plan->t3P.D2, d_plan->t3P.D3});
			}
		}

		// Compute deconv on CPU to save GPU memory and for simplicity
		std::vector<FLT> phiHatk1, phiHatk2, phiHatk3;
		std::vector<FLT> h_s, h_t, h_u;
		std::vector<std::complex<FLT>> deconv(N);

		h_s.resize(N);
		phiHatk1.resize(N);
		checkCudaErrors(cudaMemcpy(h_s.data(), d_plan->s, N * sizeof(FLT),cudaMemcpyDeviceToHost)); // nuft of shifted input
		onedim_nuft_kernel(N, h_s.data(), phiHatk1.data(), d_plan->spopts);         // fill phiHat1
		checkCudaErrors(cudaMemcpy(h_s.data(), d_s, N * sizeof(FLT),cudaMemcpyDeviceToHost)); // deconv requires original

		if(d_plan->dim > 1) {
			h_t.resize(N);
			phiHatk2.resize(N);
			checkCudaErrors(cudaMemcpy(h_t.data(), d_plan->t, N * sizeof(FLT),cudaMemcpyDeviceToHost));
			onedim_nuft_kernel(N, h_t.data(), phiHatk2.data(), d_plan->spopts);         // fill phiHat2
			checkCudaErrors(cudaMemcpy(h_t.data(), d_t, N * sizeof(FLT),cudaMemcpyDeviceToHost));
		}

		if(d_plan->dim > 2) {
			h_u.resize(N);
			phiHatk3.resize(N);
			checkCudaErrors(cudaMemcpy(h_u.data(), d_plan->u, N * sizeof(FLT),cudaMemcpyDeviceToHost));
			onedim_nuft_kernel(N, h_u.data(), phiHatk3.data(), d_plan->spopts);         // fill phiHat1
			checkCudaErrors(cudaMemcpy(h_u.data(), d_u, N * sizeof(FLT),cudaMemcpyDeviceToHost));
		}

		int Cfinite = isfinite(d_plan->t3P.C1) && isfinite(d_plan->t3P.C2) && isfinite(d_plan->t3P.C3);    // C can be nan or inf if M=0, no input NU pts
		int Cnonzero = d_plan->t3P.C1!=0.0 || d_plan->t3P.C2!=0.0 || d_plan->t3P.C3!=0.0;  // cen
#pragma omp parallel for schedule(static)
		for (BIGINT k=0;k<N;++k) {         // .... loop over NU targ freqs
			FLT phiHat = phiHatk1[k];
			if (d_plan->dim>1)
				phiHat *= phiHatk2[k];
			if (d_plan->dim>2)
				phiHat *= phiHatk3[k];
			if (Cfinite && Cnonzero) {
					FLT phase = (h_s[k] - d_plan->t3P.D1) * d_plan->t3P.C1;
				if (d_plan->dim>1)
					phase += (h_t[k] - d_plan->t3P.D2) * d_plan->t3P.C2;
				if (d_plan->dim>2)
					phase += (h_u[k] - d_plan->t3P.D3) * d_plan->t3P.C3;
				deconv[k] = std::exp(std::complex<FLT>(0, imasign * phase)) / phiHat;
			} else {
				deconv[k] = 1.0 / phiHat;
			}
		}
		checkCudaErrors(cudaMemcpy(d_plan->deconv, deconv.data(), N * sizeof(CUCPX),cudaMemcpyHostToDevice));


		// set up internal t2 plan
		int t2nmodes[] = {d_plan->nf1,d_plan->nf2,d_plan->nf3};   // t2 input is actually fw
		int ier = CUFINUFFT_MAKEPLAN(2, d_plan->dim, t2nmodes, d_plan->iflag, d_plan->maxbatchsize, d_plan->tol,
				d_plan->maxbatchsize, &(d_plan->innert2plan), NULL);
		if (ier>1) {     // if merely warning, still proceed
			fprintf(stderr,"[%s t3]: inner type 2 plan creation failed with ier=%d!\n",__func__,ier);
			return ier;
		}

		ier = CUFINUFFT_SETPTS(N, d_plan->s, d_plan->t, d_plan->u, 0, NULL, NULL, NULL, d_plan->innert2plan);  // note N = # output points (not M)
		if (ier>1) {
			fprintf(stderr,"[%s t3]: inner type 2 setpts failed, ier=%d!\n",__func__,ier);
			return ier;
		}

	}

	int nf1 = d_plan->nf1;
	int nf2 = d_plan->nf2;
	int nf3 = d_plan->nf3;


#ifdef INFO
	printf("[info  ] 2d1: (ms,mt)=(%d,%d) (nf1, nf2, nf3)=(%d,%d,%d) M=%d, ntransform = %d\n",
		d_plan->ms, d_plan->mt, d_plan->nf1, d_plan->nf2, nf3, d_plan->M,
		d_plan->ntransf);
#endif
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	switch(d_plan->dim)
	{
		case 1:
		{
			ier = ALLOCGPUMEM1D_NUPTS(d_plan);
		}
		break;
		case 2:
		{
			ier = ALLOCGPUMEM2D_NUPTS(d_plan);
		}
		break;
		case 3:
		{
			ier = ALLOCGPUMEM3D_NUPTS(d_plan);
		}
		break;
	}
#ifdef TIME
	float milliseconds = 0;
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tAllocate GPU memory NUpts%.3g s\n", milliseconds/1000);
#endif

	if(d_plan->type != 3) {
		d_plan->kx = d_kx;
		if(dim > 1)
			d_plan->ky = d_ky;
		if(dim > 2)
			d_plan->kz = d_kz;
	}

	cudaEventRecord(start);
	switch(d_plan->dim)
	{
		case 1:
		{
			if(d_plan->opts.gpu_method==1){
				ier = CUSPREAD1D_NUPTSDRIVEN_PROP(nf1,M,d_plan);
				if(ier != 0 ){
					printf("error: cuspread1d_nupts_prop, method(%d)\n",
						d_plan->opts.gpu_method);

					// Multi-GPU support: reset the device ID
					cudaSetDevice(orig_gpu_device_id);

					return 1;
				}
			}
			if(d_plan->opts.gpu_method==2){
				ier = CUSPREAD1D_SUBPROB_PROP(nf1,M,d_plan);
				if(ier != 0 ){
					printf("error: cuspread1d_subprob_prop, method(%d)\n",
						d_plan->opts.gpu_method);

					// Multi-GPU support: reset the device ID
					cudaSetDevice(orig_gpu_device_id);

					return 1;
				}
			}
		}
		break;
		case 2:
		{
			if(d_plan->opts.gpu_method==1){
				ier = CUSPREAD2D_NUPTSDRIVEN_PROP(nf1,nf2,M,d_plan);
				if(ier != 0 ){
					printf("error: cuspread2d_nupts_prop, method(%d)\n",
						d_plan->opts.gpu_method);

					// Multi-GPU support: reset the device ID
					cudaSetDevice(orig_gpu_device_id);

					return 1;
				}
			}
			if(d_plan->opts.gpu_method==2){
				ier = CUSPREAD2D_SUBPROB_PROP(nf1,nf2,M,d_plan);
				if(ier != 0 ){
					printf("error: cuspread2d_subprob_prop, method(%d)\n",
						d_plan->opts.gpu_method);

					// Multi-GPU support: reset the device ID
					cudaSetDevice(orig_gpu_device_id);

					return 1;
				}
			}
			if(d_plan->opts.gpu_method==3){
				int ier = CUSPREAD2D_PAUL_PROP(nf1,nf2,M,d_plan);
				if(ier != 0 ){
					printf("error: cuspread2d_paul_prop, method(%d)\n",
						d_plan->opts.gpu_method);

					// Multi-GPU support: reset the device ID
					cudaSetDevice(orig_gpu_device_id);

					return 1;
				}
			}
		}
		break;
		case 3:
		{
			if(d_plan->opts.gpu_method==4){
				int ier = CUSPREAD3D_BLOCKGATHER_PROP(nf1,nf2,nf3,M,d_plan);
				if(ier != 0 ){
					printf("error: cuspread3d_blockgather_prop, method(%d)\n",
						d_plan->opts.gpu_method);

					// Multi-GPU support: reset the device ID
					cudaSetDevice(orig_gpu_device_id);

					return ier;
				}
			}
			if(d_plan->opts.gpu_method==1){
				ier = CUSPREAD3D_NUPTSDRIVEN_PROP(nf1,nf2,nf3,M,d_plan);
				if(ier != 0 ){
					printf("error: cuspread3d_nuptsdriven_prop, method(%d)\n",
						d_plan->opts.gpu_method);

					// Multi-GPU support: reset the device ID
					cudaSetDevice(orig_gpu_device_id);

					return ier;
				}
			}
			if(d_plan->opts.gpu_method==2){
				int ier = CUSPREAD3D_SUBPROB_PROP(nf1,nf2,nf3,M,d_plan);
				if(ier != 0 ){
					printf("error: cuspread3d_subprob_prop, method(%d)\n",
						d_plan->opts.gpu_method);

					// Multi-GPU support: reset the device ID
					cudaSetDevice(orig_gpu_device_id);

					return ier;
				}
			}
		}
		break;
	}
#ifdef TIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tSetup Subprob properties %.3g s\n",
		milliseconds/1000);
#endif

        // Multi-GPU support: reset the device ID
        cudaSetDevice(orig_gpu_device_id);

	return 0;
}

int CUFINUFFT_EXECUTE(CUCPX* d_c, CUCPX* d_fk, CUFINUFFT_PLAN d_plan)
/*
	"exec" stage (single and double precision versions).

	The actual transformation is done here. Type and dimension of the
	transformation are defined in d_plan in previous stages.

        See ../docs/cppdoc.md for main user-facing documentation.

	Input/Output:
	d_c   a size d_plan->M CPX array on gpu (input for Type 1; output for Type
	      2)
	d_fk  a size d_plan->ms*d_plan->mt*d_plan->mu CPX array on gpu ((input for
	      Type 2; output for Type 1)

	Notes:
        i) Here CPX is a defined type meaning either complex<float> or complex<double>
	    to match the precision of the library called.
        ii) All operations are done on the GPU device (hence the d_* names)

	Melody Shih 07/25/19; Barnett 2/16/21.
*/
{
        // Mult-GPU support: set the CUDA Device ID:
        int orig_gpu_device_id;
        cudaGetDevice(& orig_gpu_device_id);
        cudaSetDevice(d_plan->opts.gpu_device_id);

	int ier;
	int type=d_plan->type;
	switch(d_plan->dim)
	{
		case 1:
		{
			if(type == 1)
				ier = CUFINUFFT1D1_EXEC(d_c, d_fk, d_plan);
			if(type == 2)
				ier = CUFINUFFT1D2_EXEC(d_c, d_fk, d_plan);
			if(type == 3){
				cerr<<"Not Implemented yet"<<endl;
				ier = 1;
			}
		}
		break;
		case 2:
		{
			if(type == 1)
				ier = CUFINUFFT2D1_EXEC(d_c, d_fk, d_plan);
			if(type == 2)
				ier = CUFINUFFT2D2_EXEC(d_c, d_fk, d_plan);
			if(type == 3){
				cerr<<"Not Implemented yet"<<endl;
				ier = 1;
			}
		}
		break;
		case 3:
		{
			if(type == 1)
				ier = CUFINUFFT3D1_EXEC(d_c, d_fk, d_plan);
			if(type == 2)
				ier = CUFINUFFT3D2_EXEC(d_c, d_fk, d_plan);
			if(type == 3){
				ier = CUFINUFFT3D3_EXEC(d_c,  d_fk, d_plan);
			}
		}
		break;
	}

        // Multi-GPU support: reset the device ID
        cudaSetDevice(orig_gpu_device_id);

	return ier;
}

int CUFINUFFT_DESTROY(CUFINUFFT_PLAN d_plan)
/*
	"destroy" stage (single and double precision versions).

	In this stage, we
		(1) free all the memories that have been allocated on gpu
		(2) delete the cuFFT plan

        Also see ../docs/cppdoc.md for main user-facing documentation.
*/
{
        // Mult-GPU support: set the CUDA Device ID:
        int orig_gpu_device_id;
        cudaGetDevice(& orig_gpu_device_id);
        cudaSetDevice(d_plan->opts.gpu_device_id);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	// Can't destroy a Null pointer.
	if(!d_plan) {
                // Multi-GPU support: reset the device ID
                cudaSetDevice(orig_gpu_device_id);
		return 1;
        }

	if(d_plan->fftplan)
		cufftDestroy(d_plan->fftplan);

	if(d_plan->type == 3) {
		CUFINUFFT_DESTROY(d_plan->innert2plan);
		checkCudaErrors(cudaFree(d_plan->cpbatch));
		checkCudaErrors(cudaFree(d_plan->prephase));
		checkCudaErrors(cudaFree(d_plan->deconv));
		checkCudaErrors(cudaFree(d_plan->kx));
		if(d_plan->dim > 1)
			checkCudaErrors(cudaFree(d_plan->ky));
		if(d_plan->dim > 2)
			checkCudaErrors(cudaFree(d_plan->kz));
		checkCudaErrors(cudaFree(d_plan->s));
		if(d_plan->dim > 1)
			checkCudaErrors(cudaFree(d_plan->t));
		if(d_plan->dim > 2)
			checkCudaErrors(cudaFree(d_plan->u));
	}

	switch(d_plan->dim)
	{
		case 1:
		{
			FREEGPUMEMORY1D(d_plan);
		}
		break;
		case 2:
		{
			FREEGPUMEMORY2D(d_plan);
		}
		break;
		case 3:
		{
			FREEGPUMEMORY3D(d_plan);
		}
		break;
	}
#ifdef TIME
	float milliseconds = 0;
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tFree gpu memory\t\t %.3g s\n", milliseconds/1000);
#endif

	/* free/destruct the plan */
	delete d_plan;
	/* set pointer to NULL now that we've hopefully free'd the memory. */
	d_plan = NULL;

        // Multi-GPU support: reset the device ID
        cudaSetDevice(orig_gpu_device_id);
	return 0;
}

int CUFINUFFT_DEFAULT_OPTS(int type, int dim, cufinufft_opts *opts)
/*
	Sets the default options in cufinufft_opts. This must be called
	before the user changes any options from default values.
	The resulting struct may then be passed (instead of NULL) to the last
	argument of cufinufft_plan().

	Options with prefix "gpu_" are used for gpu code.

	Notes:
	Values set in this function for different type and dimensions are preferable
	based on experiments. User can experiement with different settings by
	replacing them after calling this function.

	Melody Shih 07/25/19; Barnett 2/5/21.
*/
{
	int ier;
	opts->upsampfac = (FLT)2.0;

	/* following options are for gpu */
	opts->gpu_nstreams = 0;
	opts->gpu_sort = 1; // access nupts in an ordered way for nupts driven method

	opts->gpu_maxsubprobsize = 1024;
	opts->gpu_obinsizex = -1;
	opts->gpu_obinsizey = -1;
	opts->gpu_obinsizez = -1;

	opts->gpu_binsizex = -1;
	opts->gpu_binsizey = -1;
	opts->gpu_binsizez = -1;

	opts->gpu_spreadinterponly = 0; // default to do the whole nufft

	switch(dim)
	{
		case 1:
		{
			opts->gpu_kerevalmeth = 0; // using exp(sqrt())
			if(type == 1){
				opts->gpu_method = 2;
			}
			if(type == 2){
				opts->gpu_method = 1;
			}
			if(type == 3){
				cerr<<"Not Implemented yet"<<endl;
				ier = 1;
				return ier;
			}
		}
		break;
		case 2:
		{
			opts->gpu_kerevalmeth = 0; // using exp(sqrt())
			if(type == 1){
				opts->gpu_method = 2;
			}
			if(type == 2){
				opts->gpu_method = 1;
			}
			if(type == 3){
				cerr<<"Not Implemented yet"<<endl;
				ier = 1;
				return ier;
			}
		}
		break;
		case 3:
		{
			opts->gpu_kerevalmeth = 0; // using exp(sqrt())
			if(type == 1){
				opts->gpu_method = 2;
			}
			if(type == 2){
				opts->gpu_method = 1;
			}
			if(type == 3){
				opts->gpu_method = 2;
			}
		}
		break;
	}

        // By default, only use device 0
        opts->gpu_device_id = 0;

	return 0;
}
#ifdef __cplusplus
}
#endif
