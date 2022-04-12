#include <iostream>
#include <iomanip>
#include <math.h>
#include "cuda_hip_wrapper.h"
#include <helper_cuda.h>
#include <complex>

#include <cufinufft_eitherprec.h>

#include "../contrib/utils.h"

using namespace std;

int main(int argc, char* argv[])
{
	int N1, M, N;
	if (argc<3) {
		fprintf(stderr,
			"Usage: cufinufft1d1_test method N1 [M [tol]]\n"
			"Arguments:\n"
			"  method: One of\n"
			"    1: nupts driven, or\n"
			"    2: sub-problem\n"
			"  N1: The size of the 1D array.\n"
			"  M: The number of non-uniform points (default N1).\n"
			"  tol: NUFFT tolerance (default 1e-6).\n");
		return 1;
	}
	double w;
	int method;
	sscanf(argv[1],"%d",&method);
	sscanf(argv[2],"%lf",&w); N1 = (int)w;  // so can read 1e6 right!
	N = N1;
	M = N1;// let density always be 1
	if(argc>3){
		sscanf(argv[3],"%lf",&w); M  = (int)w;  // so can read 1e6 right!
	}

	FLT tol=1e-6;
	if(argc>4){
		sscanf(argv[4],"%lf",&w); tol  = (FLT)w;  // so can read 1e6 right!
	}
	int iflag=1;


	cout<<scientific<<setprecision(3);
	int ier;


	FLT *x;
	CPX *c, *fk;
	cudaMallocHost(&x, M*sizeof(FLT));
	cudaMallocHost(&c, M*sizeof(CPX));
	cudaMallocHost(&fk,N1*sizeof(CPX));

	FLT *d_x;
	CUCPX *d_c, *d_fk;
	checkCudaErrors(cudaMalloc(&d_x,M*sizeof(FLT)));
	checkCudaErrors(cudaMalloc(&d_c,M*sizeof(CUCPX)));
	checkCudaErrors(cudaMalloc(&d_fk,N1*sizeof(CUCPX)));

	// Making data
	for (int i = 0; i < M; i++) {
		x[i] = M_PI*randm11();// x in [-pi,pi)
		c[i].real(randm11());
		c[i].imag(randm11());
	}

	checkCudaErrors(cudaMemcpy(d_x,x,M*sizeof(FLT),cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_c,c,M*sizeof(CPX),cudaMemcpyHostToDevice));

	cudaEvent_t start, stop;
	float milliseconds = 0;
	float totaltime = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// warm up CUFFT (is slow, takes around 0.2 sec... )
	cudaEventRecord(start);
 	{
		int nf1=1;
		cufftHandle fftplan;
		cufftPlan1d(&fftplan,nf1,CUFFT_TYPE,1);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] dummy warmup call to CUFFT\t %.3g s\n", milliseconds/1000);

	// now to our tests...
	CUFINUFFT_PLAN dplan;
	int dim = 1;
	int type = 1;

	// Here we setup our own opts, for gpu_method.
	cufinufft_opts opts;
	ier=CUFINUFFT_DEFAULT_OPTS(type, dim, &opts);
	if(ier!=0){
	  printf("err %d: CUFINUFFT_DEFAULT_OPTS\n", ier);
	  return ier;
	}

	opts.gpu_method=method;

	int nmodes[3];
	int ntransf = 1;
	int maxbatchsize = 1;
	nmodes[0] = N1;
	nmodes[1] = 1;
	nmodes[2] = 1;
	cudaEventRecord(start);
	ier=CUFINUFFT_MAKEPLAN(type, dim, nmodes, iflag, ntransf, tol,
			       maxbatchsize, &dplan, &opts);
	if (ier!=0){
	  printf("err: cufinufft1d_plan\n");
	  return ier;
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	totaltime += milliseconds;
	printf("[time  ] cufinufft plan:\t\t %.3g s\n", milliseconds/1000);


	cudaEventRecord(start);
	ier=CUFINUFFT_SETPTS(M, d_x, NULL, NULL, 0, NULL, NULL, NULL, dplan);
	if (ier!=0){
	  printf("err: cufinufft_setpts\n");
	  return ier;
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	totaltime += milliseconds;
	printf("[time  ] cufinufft setNUpts:\t\t %.3g s\n", milliseconds/1000);


	cudaEventRecord(start);
	ier=CUFINUFFT_EXECUTE(d_c, d_fk, dplan);
	if (ier!=0){
	  printf("err: cufinufft1d1_exec\n");
	  return ier;
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	totaltime += milliseconds;
	float exec_ms = milliseconds;
	printf("[time  ] cufinufft exec:\t\t %.3g s\n", milliseconds/1000);

	cudaEventRecord(start);
	ier=CUFINUFFT_DESTROY(dplan);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	totaltime += milliseconds;
	printf("[time  ] cufinufft destroy:\t\t %.3g s\n", milliseconds/1000);

	checkCudaErrors(cudaMemcpy(fk,d_fk,N1*sizeof(CUCPX), cudaMemcpyDeviceToHost));

	printf("[Method %d] %d NU pts to %d U pts in %.3g s:      %.3g NU pts/s\n",
			opts.gpu_method,M,N1,totaltime/1000,M/totaltime*1000);
	printf("\t\t\t\t\t(exec-only thoughput: %.3g NU pts/s)\n",M/exec_ms*1000);

	int nt1 = (int)(0.37*N1);  // choose some mode index to check
	CPX Ft = CPX(0,0), J = IMA*(FLT)iflag;
	for (int j=0; j<M; ++j)
		Ft += c[j] * exp(J*(nt1*x[j]));   // crude direct
	int it = N1/2+nt1;   // index in complex F as 1d array
//	printf("[gpu   ] one mode: abs err in F[%ld is %.3g\n",(int)nt1,abs(Ft-fk[it]));
	printf("[gpu   ] one mode: rel err in F[%ld] is %.3g\n",(int)nt1,abs(Ft-fk[it])/infnorm(N,fk));

	cudaFreeHost(x);
	cudaFreeHost(c);
	cudaFreeHost(fk);
	cudaFree(d_x);
	cudaFree(d_c);
	cudaFree(d_fk);
	return 0;
}
