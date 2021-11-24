#include <iostream>
#include <iomanip>
#include <math.h>
#include <helper_cuda.h>
#include <complex>
#include <profile.h>

#include <cufinufft_eitherprec.h>

#include "../contrib/utils.h"

using namespace std;

int main(int argc, char* argv[])
{
	int N, M;
	if (argc<3) {
		fprintf(stderr,
			"Usage: cufinufft3d2_test method N1 N2 N3 [M [tol]]\n"
			"Arguments:\n"
			"  method: One of\n"
			"    1: nupts driven, or\n"
			"    2: sub-problem.\n"
			"  M: The number of non-uniform input points.\n"
			"  N: The number of non-uniform output points.\n"
			"  tol: NUFFT tolerance (default 1e-6).\n");
		return 1;
	}
	double w;
	int method;
	sscanf(argv[1],"%d",&method);
	sscanf(argv[2],"%lf",&w); M = (int)w;  // so can read 1e6 right!
	sscanf(argv[3],"%lf",&w); N = (int)w;  // so can read 1e6 right!

	FLT tol=1e-6;
	if(argc>4){
		sscanf(argv[4],"%lf",&w); tol  = (FLT)w;  // so can read 1e6 right!
	}
	int iflag=1;


	cout<<scientific<<setprecision(3);
	int ier;


	FLT *x, *y, *z;
	cudaMallocHost(&x, M*sizeof(FLT));
	cudaMallocHost(&y, M*sizeof(FLT));
	cudaMallocHost(&z, M*sizeof(FLT));

	FLT *s, *t, *u;
	cudaMallocHost(&s, N*sizeof(FLT));
	cudaMallocHost(&t, N*sizeof(FLT));
	cudaMallocHost(&u, N*sizeof(FLT));

	CPX *c, *fk;
	cudaMallocHost(&c, M*sizeof(CPX));
	cudaMallocHost(&fk,N*sizeof(CPX));

	FLT *d_x, *d_y, *d_z;
	checkCudaErrors(cudaMalloc(&d_x,M*sizeof(FLT)));
	checkCudaErrors(cudaMalloc(&d_y,M*sizeof(FLT)));
	checkCudaErrors(cudaMalloc(&d_z,M*sizeof(FLT)));

	FLT *d_s, *d_t, *d_u;
	checkCudaErrors(cudaMalloc(&d_s,N*sizeof(FLT)));
	checkCudaErrors(cudaMalloc(&d_t,N*sizeof(FLT)));
	checkCudaErrors(cudaMalloc(&d_u,N*sizeof(FLT)));

	CUCPX *d_c, *d_fk;
	checkCudaErrors(cudaMalloc(&d_c,M*sizeof(CUCPX)));
	checkCudaErrors(cudaMalloc(&d_fk,N*sizeof(CUCPX)));

	// Making data
	for (int i = 0; i < M; i++) {
		x[i] = M_PI*randm11();// x in [-pi,pi)
		y[i] = M_PI*randm11();
		z[i] = M_PI*randm11();
	}

	for (int i = 0; i < N; i++) {
		s[i] = M_PI*randm11();// x in [-pi,pi)
		t[i] = M_PI*randm11();
		u[i] = M_PI*randm11();
	}

	for(int i=0; i<M; i++){
		c[i].real(randm11());
		c[i].imag(randm11());
	}

	checkCudaErrors(cudaMemcpy(d_x,x,M*sizeof(FLT),cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_y,y,M*sizeof(FLT),cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_z,z,M*sizeof(FLT),cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_s,s,N*sizeof(FLT),cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_t,t,N*sizeof(FLT),cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_u,u,N*sizeof(FLT),cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_c,c,M*sizeof(CPX),
		cudaMemcpyHostToDevice));

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

        // now to the test...
	CUFINUFFT_PLAN dplan;
	int dim = 3;
	int type = 3;

	// Here we setup our own opts, for gpu_method.
	cufinufft_opts opts;
	ier=CUFINUFFT_DEFAULT_OPTS(type, dim, &opts);
	if(ier!=0){
	  printf("err %d: CUFINUFFT_DEFAULT_OPTS\n", ier);
	  return ier;
	}
	opts.gpu_method=method;

	int ntransf = 1;
	int maxbatchsize = 1;

	cudaEventRecord(start);
	{
		PROFILE_CUDA_GROUP("cufinufft3d_plan",2);
		ier=CUFINUFFT_MAKEPLAN(type, dim, NULL, iflag, ntransf, tol,
				       maxbatchsize, &dplan, &opts);
		if (ier!=0){
			printf("err: cufinufft_makeplan\n");
		}
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	totaltime += milliseconds;
	printf("[time  ] cufinufft plan:\t\t %.3g s\n", milliseconds/1000);

	cudaEventRecord(start);
	{
		PROFILE_CUDA_GROUP("cufinufft_setpts",3);
		ier=CUFINUFFT_SETPTS(M, d_x, d_y, d_z, N, d_s, d_t, d_u, dplan);
		if (ier!=0){
		  printf("err: cufinufft_setpts\n");
		  return ier;
		}
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	totaltime += milliseconds;
	printf("[time  ] cufinufft setNUpts:\t\t %.3g s\n", milliseconds/1000);

	cudaEventRecord(start);
	{
		PROFILE_CUDA_GROUP("cufinufft_execute",4);
		ier=CUFINUFFT_EXECUTE(d_c, d_fk, dplan);
		if (ier!=0){
		  printf("err: cufinufft_execute\n");
		  return ier;
		}
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	totaltime += milliseconds;
	float exec_ms =	milliseconds;
	printf("[time  ] cufinufft exec:\t\t %.3g s\n", milliseconds/1000);

	cudaEventRecord(start);
	{
		PROFILE_CUDA_GROUP("cufinufft3d_destroy",5);
		ier=CUFINUFFT_DESTROY(dplan);
		if (ier!=0){
		  printf("err: cufinufft_destroy\n");
		  return ier;
		}
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	totaltime += milliseconds;
	printf("[time  ] cufinufft destroy:\t\t %.3g s\n", milliseconds/1000);

	checkCudaErrors(cudaMemcpy(fk,d_fk,N*sizeof(CUCPX),cudaMemcpyDeviceToHost));

	printf("[Method %d] %ld U pts to %d NU pts in %.3g s:\t%.3g NU pts/s\n",
			opts.gpu_method, N ,M,totaltime/1000,M/totaltime*1000);
        printf("\t\t\t\t\t(exec-only thoughput: %.3g NU pts/s)\n",M/exec_ms*1000);

	int jt = N/2;          // check arbitrary choice of one targ pt
	CPX J = IMA*(FLT)iflag;
	CPX ft = CPX(0,0);
	for (int i = 0; i < M; ++i)
	    ft += c[i] * exp(J * (x[i] * s[jt] + y[i] * t[jt] + z[i] * u[jt]));   // crude direct
	printf("[gpu   ] one targ: rel err in c[%ld] is %.3g\n",(int64_t)jt,
		abs(fk[jt]-ft)/infnorm(N,fk));

	cudaFreeHost(x);
	cudaFreeHost(y);
	cudaFreeHost(z);
	cudaFreeHost(s);
	cudaFreeHost(y);
	cudaFreeHost(u);
	cudaFreeHost(c);
	cudaFreeHost(fk);
	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(d_z);
	cudaFree(d_s);
	cudaFree(d_t);
	cudaFree(d_u);
	cudaFree(d_c);
	cudaFree(d_fk);
	return 0;
}
