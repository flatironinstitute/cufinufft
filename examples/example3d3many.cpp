/* This is an example of performing 3d3many
   in single precision.
*/


#include <iostream>
#include <iomanip>
#include <math.h>
#include <complex>
#include <random>

#include <cufinufft.h>
#include <vector>

using namespace std;

int main(int argc, char* argv[])
/*
 * example code for 3D Type 3 transformation.
 *
 * To compile the code:
 * nvcc example3d3many.cpp -o example3d3many -I/loc/to/cufinufft/include /loc/to/cufinufft/lib-static/libcufinufft.a -lcudart -lcufft -lnvToolsExt
 *
 * or
 * export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/loc/to/cufinufft/lib
 * nvcc example3d3many.cpp -o example3d3many -I/loc/to/cufinufft/include -L/loc/to/cufinufft/lib/ -lcufinufft
 *
 *
 */
{
	cout<<scientific<<setprecision(3);
  std::minstd_rand rand_gen(42);
  std::uniform_real_distribution<double> rand_dist(-M_PI, M_PI);

	int ier;
	int M = 65536;
	int N = 16384;
	int ntransf = 1;
	int maxbatchsize = 1;
	int iflag=1;
	double tol=1e-3;

	// X Y Z
	double *x, *y, *z;
	cudaMallocHost(&x, M*sizeof(double));
	cudaMallocHost(&y, M*sizeof(double));
	cudaMallocHost(&z, M*sizeof(double));

	double *d_x, *d_y, *d_z;
	cudaMalloc(&d_x,M*sizeof(double));
	cudaMalloc(&d_y,M*sizeof(double));
	cudaMalloc(&d_z,M*sizeof(double));

	for (int i=0; i<M; i++) {
		x[i] = rand_dist(rand_gen);
		y[i] = rand_dist(rand_gen);
		z[i] = rand_dist(rand_gen);
	}

	// S T U
	double *s, *t, *u;
	cudaMallocHost(&s, N*sizeof(double));
	cudaMallocHost(&t, N*sizeof(double));
	cudaMallocHost(&u, N*sizeof(double));

	double *d_s, *d_t, *d_u;
	cudaMalloc(&d_s,N*sizeof(double));
	cudaMalloc(&d_t,N*sizeof(double));
	cudaMalloc(&d_u,N*sizeof(double));

	for (int i=0; i<N; i++) {
		s[i] = rand_dist(rand_gen);
		t[i] = rand_dist(rand_gen);
		u[i] = rand_dist(rand_gen);
	}


	cudaMemcpy(d_s,s,N*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(d_t,t,N*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(d_u,u,N*sizeof(double),cudaMemcpyHostToDevice);


	// Input / output
	complex<double> *c, *fk;
	cuDoubleComplex *d_c, *d_fk;
	cudaMallocHost(&c, M*ntransf*sizeof(complex<double>));
	cudaMallocHost(&fk, N*ntransf*sizeof(complex<double>));
	cudaMalloc(&d_c,M*ntransf*sizeof(cuDoubleComplex));
	cudaMalloc(&d_fk,N*ntransf*sizeof(cuDoubleComplex));

	for(int i=0; i<M*ntransf; i++){
		c[i] = rand_dist(rand_gen);
		c[i] += std::complex<double>(0, rand_dist(rand_gen));
	}
	cudaMemcpy(d_c,c,M*ntransf*sizeof(cuDoubleComplex),cudaMemcpyHostToDevice);

	cudaMemset(d_fk, 0, N*ntransf*sizeof(cuDoubleComplex));


	cudaMemcpy(d_x,x,M*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(d_y,y,M*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(d_z,z,M*sizeof(double),cudaMemcpyHostToDevice);

	// Create plan
	cufinufft_plan dplan;

	int dim = 3;
	int type = 3;


	ier=cufinufft_makeplan(type, dim, NULL, iflag, ntransf, tol,
				maxbatchsize, &dplan, NULL);

	ier=cufinufft_setpts(M, d_x, d_y, d_z, N, d_s, d_t, d_u, dplan);

	ier=cufinufft_execute(d_c, d_fk, dplan);

	ier=cufinufft_destroy(dplan);

	cudaMemcpy(fk,d_fk,N*ntransf*sizeof(cuDoubleComplex),cudaMemcpyDeviceToHost);


	cout<<endl<<"Accuracy check:"<<endl;
	int jt = N/2;          // check arbitrary choice of one targ pt
	complex<double> J(0, iflag);
	for(int n=0; n<ntransf; n+=1){
		complex<double> ft(0,0);
		complex<double>* fkstart = fk + n*N;
		complex<double>* cstart = c + n*M;

		for (int i = 0; i < M; ++i)
				ft += cstart[i] * exp(J * (x[i] * s[jt] + y[i] * t[jt] + z[i] * u[jt]));   // crude direct

		printf("[gpu   ] one targ: rel err in F[%ld] is %.3g\n",(int64_t)jt,
				abs(fkstart[jt]-ft)/infnorm(N,fk));
	}

	cudaFreeHost(x);
	cudaFreeHost(y);
	cudaFreeHost(z);
	cudaFreeHost(s);
	cudaFreeHost(t);
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
