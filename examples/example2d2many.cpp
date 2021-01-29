/* This is an example of performing 2d2many
   in double precision.
*/

#include <iostream>
#include <iomanip>
#include <math.h>
#include <complex>

#include <cufinufft.h>

using namespace std;

int main(int argc, char* argv[])
/*
 * example code for 2D Type 1 transformation.
 *
 * To compile the code:
 * nvcc example2d2many.cpp -o example2d2many loc/to/cufinufft/lib-static/libcufinufft.a -I/loc/to/cufinufft/include -lcudart -lcufft -lnvToolsExt
 *
 * or
 * export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/loc/to/cufinufft/lib
 * nvcc -DSINGLE example2d2many.cpp -L/loc/to/cufinufft/lib/ -I/loc/to/cufinufft/include -o example2d1 -lcufinufft
 *
 *
 */
{
	cout<<scientific<<setprecision(3);

	int ier;
	int N1 = 128;
	int N2 = 128;
	int M = 10;
	int ntransf = 4;
	int maxbatchsize = 4;
	int iflag=1;
	double tol=1e-6;

	double *x, *y;
	complex<double> *c, *fk;
	cudaMallocHost(&x, M*sizeof(double));
	cudaMallocHost(&y, M*sizeof(double));
	cudaMallocHost(&c, M*ntransf*sizeof(complex<double>));
	cudaMallocHost(&fk,N1*N2*ntransf*sizeof(complex<double>));

	double *d_x, *d_y;
	cuDoubleComplex *d_c, *d_fk;
	cudaMalloc(&d_x,M*sizeof(double));
	cudaMalloc(&d_y,M*sizeof(double));
	cudaMalloc(&d_c,M*ntransf*sizeof(cuDoubleComplex));
	cudaMalloc(&d_fk,N1*N2*ntransf*sizeof(cuDoubleComplex));

	for (int i=0; i<M; i++) {
		x[i] = M_PI*randm11();
		y[i] = M_PI*randm11();
	}

	for(int i=0; i<N1*N2*ntransf; i++){
		fk[i].real(randm11());
		fk[i].imag(randm11());
	}
	cudaMemcpy(d_x,x,M*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(d_y,y,M*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(d_fk,fk,N1*N2*ntransf*sizeof(cuDoubleComplex),cudaMemcpyHostToDevice);

	cufinufft_plan dplan;

	int dim = 2;
	int nmodes[3];
	int type = 2;

	nmodes[0] = N1;
	nmodes[1] = N2;
	nmodes[2] = 1;

	ier=cufinufft_makeplan(type, dim, nmodes, iflag, ntransf, tol,
			       maxbatchsize, &dplan, NULL);

	ier=cufinufft_setpts(M, d_x, d_y, NULL, 0, NULL, NULL, NULL, dplan);

	ier=cufinufft_execute(d_c, d_fk, dplan);

	ier=cufinufft_destroy(dplan);

	cudaMemcpy(c,d_c,M*ntransf*sizeof(cuDoubleComplex),cudaMemcpyDeviceToHost);

	cout<<endl<<"Accuracy check:"<<endl;
	complex<double>* fkstart;
	complex<double>* cstart;
	for(int t=0; t<ntransf; t++){
		fkstart = fk + t*N1*N2;
		cstart = c + t*M;
		int jt = M/2;          // check arbitrary choice of one targ pt
		complex<double> J(0,iflag*1);
		complex<double> ct(0,0);
		int m=0;
		for (int m2=-(N2/2); m2<=(N2-1)/2; ++m2)  // loop in correct order over F
			for (int m1=-(N1/2); m1<=(N1-1)/2; ++m1)
				ct += fkstart[m++] * exp(J*(m1*x[jt] + m2*y[jt]));   // crude direct

		printf("[gpu %3d] one targ: rel err in c[%ld] is %.3g\n",t,(int)jt,abs(cstart[jt]-ct)/infnorm(M,c));
	}

	cudaFreeHost(x);
	cudaFreeHost(y);
	cudaFreeHost(c);
	cudaFreeHost(fk);

	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(d_c);
	cudaFree(d_fk);
	return 0;
}
