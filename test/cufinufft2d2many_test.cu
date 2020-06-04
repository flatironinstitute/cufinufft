#include <iostream>
#include <iomanip>
#include <math.h>
#include <helper_cuda.h>
#include <complex>

#include <cufinufft.h>
#include <profile.h>
#include "../contrib/utils.h"

using namespace std;

int main(int argc, char* argv[])
{
	int N1, N2, M;
	int ntransf, maxbatchsize;
	if (argc<4) {
		fprintf(stderr,
			"Usage: cufinufft2d2many_test method N1 N2 [ntransf [maxbatchsize [M [tol]]]]\n"
			"Arguments:\n"
			"  method: One of\n"
			"    1: nupts driven, or\n"
			"    2: sub-problem.\n"
			"  N1, N2: The size of the 2D array.\n"
			"  ntransf: Number of inputs (default 2 ^ 27 / (N1 * N2)).\n"
			"  maxbatchsize: Number of simultaneous transforms (default min(8, ntransf)).\n"
			"  M: The number of non-uniform points (default N1 * N2).\n"
			"  tol: NUFFT tolerance (default 1e-6).\n");
		return 1;
	}  
	double w;
	int method;
	sscanf(argv[1],"%d",&method);
	sscanf(argv[2],"%lf",&w); N1 = (int)w;  // so can read 1e6 right!
	sscanf(argv[3],"%lf",&w); N2 = (int)w;  // so can read 1e6 right!
	M = 2*N1*N2;// let density always be 2
	ntransf = pow(2,28)/M;
	if(argc>4){
		sscanf(argv[4],"%d",&ntransf);
	}

	maxbatchsize = min(8, ntransf);
	if(argc>5){
		sscanf(argv[5],"%d",&maxbatchsize);
	}

	if(argc>6){
		sscanf(argv[6],"%lf",&w); M  = (int)w;  // so can read 1e6 right!
	}

	FLT tol=1e-6;
	if(argc>7){
		sscanf(argv[7],"%lf",&w); tol  = (FLT)w;  // so can read 1e6 right!
	}
	int iflag=1;
	


	cout<<scientific<<setprecision(3);
	int ier;

	printf("#modes = %d, #inputs = %d, #NUpts = %d\n", N1*N2, ntransf, M);

	FLT *x, *y;
	CPX *c, *fk;
#if 1
	cudaMallocHost(&x, M*sizeof(FLT));
	cudaMallocHost(&y, M*sizeof(FLT));
	cudaMallocHost(&c, ntransf*M*sizeof(CPX));
	cudaMallocHost(&fk,ntransf*N1*N2*sizeof(CPX));
#else
	x = (FLT*) malloc(M*sizeof(FLT));
	y = (FLT*) malloc(M*sizeof(FLT));
	c = (CPX*) malloc(ntransf*M*sizeof(CPX));
	fk = (CPX*) malloc(ntransf*N1*N2*sizeof(CPX));
#endif
	FLT *d_x, *d_y;
	CUCPX *d_c, *d_fk;
	checkCudaErrors(cudaMalloc(&d_x,M*sizeof(FLT)));
	checkCudaErrors(cudaMalloc(&d_y,M*sizeof(FLT)));
	checkCudaErrors(cudaMalloc(&d_c,ntransf*M*sizeof(CUCPX)));
	checkCudaErrors(cudaMalloc(&d_fk,ntransf*N1*N2*sizeof(CUCPX)));

	// Making data
	for (int i = 0; i < M; i++) {
		x[i] = M_PI*randm11();// x in [-pi,pi)
		y[i] = M_PI*randm11();
	}

	for(int i=0; i<ntransf*N1*N2; i++){
		fk[i].real(randm11());
		fk[i].imag(randm11());
	}

	checkCudaErrors(cudaMemcpy(d_x,x,M*sizeof(FLT),cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_y,y,M*sizeof(FLT),cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_fk,fk,N1*N2*ntransf*sizeof(CUCPX),cudaMemcpyHostToDevice));

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	/*warm up gpu*/
	cudaEventRecord(start);
	{
		PROFILE_CUDA_GROUP("Warm Up",1);
		char *a;
		checkCudaErrors(cudaMalloc(&a,1));
	}
	float milliseconds = 0;
	double totaltime = 0;
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tWarm up GPU \t\t %.3g s\n", milliseconds/1000);

	cufinufft_plan dplan;
	int dim = 2;
	int type = 2;
	ier=cufinufft_default_opts(type, dim, &dplan.opts);
	dplan.opts.gpu_method=method;
	dplan.opts.gpu_kerevalmeth=1;

	int nmodes[3];
	nmodes[0] = N1;
	nmodes[1] = N2;
	nmodes[2] = 1;
	cudaEventRecord(start);
	{
		PROFILE_CUDA_GROUP("cufinufft2d_plan",2);
		ier=cufinufft_makeplan(type, dim, nmodes, iflag, ntransf, tol, 
			maxbatchsize, &dplan);
		if (ier!=0){
			printf("err: cufinufft2d_plan\n");
		}
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	totaltime += milliseconds;
	printf("[time  ] cufinufft plan:\t\t %.3g s\n", milliseconds/1000);
	cudaEventRecord(start);
	{
		PROFILE_CUDA_GROUP("cufinufft2d_setNUpts",3);
		ier=cufinufft_setNUpts(M, d_x, d_y, NULL, 0, NULL, NULL, NULL, &dplan);
		if (ier!=0){
			printf("err: cufinufft2d_setNUpts\n");
		}
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	totaltime += milliseconds;
	printf("[time  ] cufinufft setNUpts:\t\t %.3g s\n", milliseconds/1000);
	cudaEventRecord(start);
	{
		PROFILE_CUDA_GROUP("cufinufft2d_exec",4);
		ier=cufinufft_exec(d_c, d_fk, &dplan);
		if (ier!=0){
			printf("err: cufinufft2d2_exec\n");
		}
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	totaltime += milliseconds;
	printf("[time  ] cufinufft exec:\t\t %.3g s\n", milliseconds/1000);
	cudaEventRecord(start);
	{
		PROFILE_CUDA_GROUP("cufinufft2d_destroy",5);
		ier=cufinufft_destroy(&dplan);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	totaltime += milliseconds;
	printf("[time  ] cufinufft destroy:\t\t %.3g s\n", milliseconds/1000);
	// This must be here, since in gpu code, x, y gets modified if pirange=1
	checkCudaErrors(cudaMemcpy(c,d_c,M*ntransf*sizeof(CUCPX),cudaMemcpyDeviceToHost));
#if 1 
	CPX* fkstart; 
	CPX* cstart;
	for(int t=0; t<ntransf; t++){
		fkstart = fk + t*N1*N2;
		cstart = c + t*M;
		int jt = M/2;          // check arbitrary choice of one targ pt
		CPX J = IMA*(FLT)iflag;
		CPX ct = CPX(0,0);
		int m=0;
		for (int m2=-(N2/2); m2<=(N2-1)/2; ++m2)  // loop in correct order over F
			for (int m1=-(N1/2); m1<=(N1-1)/2; ++m1)
				ct += fkstart[m++] * exp(J*(m1*x[jt] + m2*y[jt]));   // crude direct
		
		printf("[gpu   ] one targ: rel err in c[%ld] is %.3g\n",(int64_t)jt,abs(cstart[jt]-ct)/infnorm(M,c));
	}
#endif
#if 0
	cout<<"[result-input]"<<endl;
	for(int j=0; j<nf2; j++){
		//        if( j % opts.gpu_binsizey == 0)
		//                printf("\n");
		for (int i=0; i<nf1; i++){
			//                if( i % opts.gpu_binsizex == 0 && i!=0)
			//                        printf(" |");
			printf(" (%2.3g,%2.3g)",fw[i+j*nf1].real(),fw[i+j*nf1].imag() );
		}
		cout<<endl;
	}
#endif	
	printf("[totaltime] %.3g us, speed %.3g NUpts/s\n", totaltime*1000, M*ntransf/totaltime*1000);
#if 1
	cudaFreeHost(x);
	cudaFreeHost(y);
	cudaFreeHost(c);
	cudaFreeHost(fk);
#else
	free(x);
	free(y);
	free(c);
	free(fk);
#endif
	checkCudaErrors(cudaDeviceReset());
	return 0;
}
