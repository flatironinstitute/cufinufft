#include <helper_cuda.h>
#include <iostream>
#include <iomanip>

#include <cuComplex.h>
#include "memtransfer.h"

using namespace std;

int allocgpumem2d_plan(cufinufft_plan *d_plan)
/* 
	wrapper for gpu memory allocation in "plan" stage.

	Melody Shih 07/25/19
*/
{
	int nf1 = d_plan->nf1;
	int nf2 = d_plan->nf2;
	int maxbatchsize = d_plan->maxbatchsize;

	d_plan->byte_now=0;
	// No extra memory is needed in nuptsdriven method (case 1)
	switch(d_plan->opts.gpu_method)
	{
		case 1:
			{
				if(d_plan->opts.gpu_sort){
					int numbins[2];
					numbins[0] = ceil((FLT) nf1/d_plan->opts.gpu_binsizex);
					numbins[1] = ceil((FLT) nf2/d_plan->opts.gpu_binsizey);
					checkCudaErrors(cudaMalloc(&d_plan->numsubprob,numbins[0]*
						numbins[1]*sizeof(int)));
					checkCudaErrors(cudaMalloc(&d_plan->binsize,numbins[0]*
						numbins[1]*sizeof(int)));
					checkCudaErrors(cudaMalloc(&d_plan->binstartpts,numbins[0]*
						numbins[1]*sizeof(int)));
				}
			}
			break;
		case 2:
			{
				int numbins[2];
				numbins[0] = ceil((FLT) nf1/d_plan->opts.gpu_binsizex);
				numbins[1] = ceil((FLT) nf2/d_plan->opts.gpu_binsizey);
				checkCudaErrors(cudaMalloc(&d_plan->numsubprob,numbins[0]*
						numbins[1]*sizeof(int)));
				checkCudaErrors(cudaMalloc(&d_plan->binsize,numbins[0]*
						numbins[1]*sizeof(int)));
				checkCudaErrors(cudaMalloc(&d_plan->binstartpts,numbins[0]*
						numbins[1]*sizeof(int)));
				checkCudaErrors(cudaMalloc(&d_plan->subprobstartpts,
						(numbins[0]*numbins[1]+1)*sizeof(int)));
			}
		case 3:
			{
				int numbins[2];
				numbins[0] = ceil((FLT) nf1/d_plan->opts.gpu_binsizex);
				numbins[1] = ceil((FLT) nf2/d_plan->opts.gpu_binsizey);
				checkCudaErrors(cudaMalloc(&d_plan->finegridsize,nf1*nf2*
						sizeof(int)));
				checkCudaErrors(cudaMalloc(&d_plan->fgstartpts,nf1*nf2*
						sizeof(int)));
				checkCudaErrors(cudaMalloc(&d_plan->numsubprob,numbins[0]*
						numbins[1]*sizeof(int)));
				checkCudaErrors(cudaMalloc(&d_plan->binsize,numbins[0]*
						numbins[1]*sizeof(int)));
				checkCudaErrors(cudaMalloc(&d_plan->binstartpts,numbins[0]*
						numbins[1]*sizeof(int)));
				checkCudaErrors(cudaMalloc(&d_plan->subprobstartpts,
						(numbins[0]*numbins[1]+1)*sizeof(int)));
			}
			break;
		default:
			cerr << "err: invalid method " << endl;
	}

	checkCudaErrors(cudaMalloc(&d_plan->fw, maxbatchsize*nf1*nf2*
			sizeof(CUCPX)));
	//checkCudaErrors(cudaMalloc(&d_plan->fk,maxbatchsize*ms*mt*
	//	sizeof(CUCPX)));
	checkCudaErrors(cudaMalloc(&d_plan->fwkerhalf1,(nf1/2+1)*sizeof(FLT)));
	checkCudaErrors(cudaMalloc(&d_plan->fwkerhalf2,(nf2/2+1)*sizeof(FLT)));

	cudaStream_t* streams =(cudaStream_t*) malloc(d_plan->opts.gpu_nstreams*
		sizeof(cudaStream_t));
	for(int i=0; i<d_plan->opts.gpu_nstreams; i++)
		checkCudaErrors(cudaStreamCreate(&streams[i]));
	d_plan->streams = streams;
	return 0;
}

int allocgpumem2d_nupts(cufinufft_plan *d_plan)
/* 
	wrapper for gpu memory allocation in "setNUpts" stage.

	Melody Shih 07/25/19
*/
{
	int M = d_plan->M;
	//int maxbatchsize = d_plan->maxbatchsize;

	//checkCudaErrors(cudaMalloc(&d_plan->kx,M*sizeof(FLT)));
	//checkCudaErrors(cudaMalloc(&d_plan->ky,M*sizeof(FLT)));
	//checkCudaErrors(cudaMalloc(&d_plan->c,maxbatchsize*M*sizeof(CUCPX)));
	switch(d_plan->opts.gpu_method)
	{
		case 1:
			{
				if(d_plan->opts.gpu_sort)
					checkCudaErrors(cudaMalloc(&d_plan->sortidx, M*sizeof(int)));
				checkCudaErrors(cudaMalloc(&d_plan->idxnupts,M*sizeof(int)));
			}
			break;
		case 2:
		case 3:
			{
				checkCudaErrors(cudaMalloc(&d_plan->idxnupts,M*sizeof(int)));
				checkCudaErrors(cudaMalloc(&d_plan->sortidx, M*sizeof(int)));
			}
			break;
		default: 
			cerr<<"err: invalid method" << endl;
	}
	return 0;
}

void freegpumemory2d(cufinufft_plan *d_plan)
/* 
	wrapper for freeing gpu memory.

	Melody Shih 07/25/19
*/
{
	checkCudaErrors(cudaFree(d_plan->fw));
	//cudaFree(d_plan->kx);
	//cudaFree(d_plan->ky);
	//cudaFree(d_plan->c);
	checkCudaErrors(cudaFree(d_plan->fwkerhalf1));
	checkCudaErrors(cudaFree(d_plan->fwkerhalf2));
	switch(d_plan->opts.gpu_method)
	{
		case 1:
			{
				if(d_plan->opts.gpu_sort){
					checkCudaErrors(cudaFree(d_plan->idxnupts));
					checkCudaErrors(cudaFree(d_plan->sortidx));
					checkCudaErrors(cudaFree(d_plan->binsize));
					checkCudaErrors(cudaFree(d_plan->binstartpts));
				}else{
					checkCudaErrors(cudaFree(d_plan->idxnupts));
				}
			}
			break;
		case 2:
			{
				checkCudaErrors(cudaFree(d_plan->idxnupts));
				checkCudaErrors(cudaFree(d_plan->sortidx));
				checkCudaErrors(cudaFree(d_plan->numsubprob));
				checkCudaErrors(cudaFree(d_plan->binsize));
				checkCudaErrors(cudaFree(d_plan->binstartpts));
				checkCudaErrors(cudaFree(d_plan->subprobstartpts));
				checkCudaErrors(cudaFree(d_plan->subprob_to_bin));
			}
			break;
		case 3:
			{
				checkCudaErrors(cudaFree(d_plan->idxnupts));
				checkCudaErrors(cudaFree(d_plan->sortidx));
				checkCudaErrors(cudaFree(d_plan->numsubprob));
				checkCudaErrors(cudaFree(d_plan->binsize));
				checkCudaErrors(cudaFree(d_plan->finegridsize));
				checkCudaErrors(cudaFree(d_plan->binstartpts));
				checkCudaErrors(cudaFree(d_plan->subprobstartpts));
				checkCudaErrors(cudaFree(d_plan->subprob_to_bin));
			}
			break;
	}

	for(int i=0; i<d_plan->opts.gpu_nstreams; i++)
		checkCudaErrors(cudaStreamDestroy(d_plan->streams[i]));
}

int allocgpumem1d_plan(cufinufft_plan *d_plan)
{
	cerr<<"Not yet implemented"<<endl;
	return 1;
}
int allocgpumem1d_nupts(cufinufft_plan *d_plan)
{
	cerr<<"Not yet implemented"<<endl;
	return 1;
}
void freegpumemory1d(cufinufft_plan *d_plan)
{
	cerr<<"Not yet implemented"<<endl;
}

int allocgpumem3d_plan(cufinufft_plan *d_plan)
/* 
	wrapper for gpu memory allocation in "plan" stage.

	Melody Shih 07/25/19
*/
{
	//int ms = d_plan->ms;
	//int mt = d_plan->mt;
	//int mu = d_plan->mu;
	int nf1 = d_plan->nf1;
	int nf2 = d_plan->nf2;
	int nf3 = d_plan->nf3;
	int maxbatchsize = d_plan->maxbatchsize;

	d_plan->byte_now=0;
	// No extra memory is needed in nuptsdriven method;
	switch(d_plan->opts.gpu_method)
	{
		case 1:
			{
				if(d_plan->opts.gpu_sort){
					int numbins[3];
					numbins[0] = ceil((FLT) nf1/d_plan->opts.gpu_binsizex);
					numbins[1] = ceil((FLT) nf2/d_plan->opts.gpu_binsizey);
					numbins[2] = ceil((FLT) nf3/d_plan->opts.gpu_binsizez);
					checkCudaErrors(cudaMalloc(&d_plan->numsubprob,numbins[0]*
						numbins[1]*numbins[2]*sizeof(int)));
					checkCudaErrors(cudaMalloc(&d_plan->binsize,numbins[0]*
						numbins[1]*numbins[2]*sizeof(int)));
					checkCudaErrors(cudaMalloc(&d_plan->binstartpts,numbins[0]*
						numbins[1]*numbins[2]*sizeof(int)));
				}
			}
			break;
		case 2:
			{
				int numbins[3];
				numbins[0] = ceil((FLT) nf1/d_plan->opts.gpu_binsizex);
				numbins[1] = ceil((FLT) nf2/d_plan->opts.gpu_binsizey);
				numbins[2] = ceil((FLT) nf3/d_plan->opts.gpu_binsizez);
				checkCudaErrors(cudaMalloc(&d_plan->numsubprob,numbins[0]*
					numbins[1]*numbins[2]*sizeof(int)));
				checkCudaErrors(cudaMalloc(&d_plan->binsize,numbins[0]*
					numbins[1]*numbins[2]*sizeof(int)));
				checkCudaErrors(cudaMalloc(&d_plan->binstartpts,numbins[0]*
					numbins[1]*numbins[2]*sizeof(int)));
				checkCudaErrors(cudaMalloc(&d_plan->subprobstartpts,
					(numbins[0]*numbins[1]*numbins[2]+1)*sizeof(int)));
			}
			break;
		case 4:
			{
				int numobins[3], numbins[3];
				int binsperobins[3];
				numobins[0] = ceil((FLT) nf1/d_plan->opts.gpu_obinsizex);
				numobins[1] = ceil((FLT) nf2/d_plan->opts.gpu_obinsizey);
				numobins[2] = ceil((FLT) nf3/d_plan->opts.gpu_obinsizez);

				binsperobins[0] = d_plan->opts.gpu_obinsizex/
					d_plan->opts.gpu_binsizex;
				binsperobins[1] = d_plan->opts.gpu_obinsizey/
					d_plan->opts.gpu_binsizey;
				binsperobins[2] = d_plan->opts.gpu_obinsizez/
					d_plan->opts.gpu_binsizez;

				numbins[0] = numobins[0]*(binsperobins[0]+2);
				numbins[1] = numobins[1]*(binsperobins[1]+2);
				numbins[2] = numobins[2]*(binsperobins[2]+2);

				checkCudaErrors(cudaMalloc(&d_plan->numsubprob,
					numobins[0]*numobins[1]*numobins[2]*sizeof(int)));
				checkCudaErrors(cudaMalloc(&d_plan->binsize,
					numbins[0]*numbins[1]*numbins[2]*sizeof(int)));
				checkCudaErrors(cudaMalloc(&d_plan->binstartpts,
					(numbins[0]*numbins[1]*numbins[2]+1)*sizeof(int)));
				checkCudaErrors(cudaMalloc(&d_plan->subprobstartpts,(numobins[0]
					*numobins[1]*numobins[2]+1)*sizeof(int)));
			}
			break;
		default:
			cerr << "err: invalid method" << endl;
	}
	checkCudaErrors(cudaMalloc(&d_plan->fw, maxbatchsize*nf1*nf2*nf3*
		sizeof(CUCPX)));

	checkCudaErrors(cudaMalloc(&d_plan->fwkerhalf1,(nf1/2+1)*sizeof(FLT)));
	checkCudaErrors(cudaMalloc(&d_plan->fwkerhalf2,(nf2/2+1)*sizeof(FLT)));
	checkCudaErrors(cudaMalloc(&d_plan->fwkerhalf3,(nf3/2+1)*sizeof(FLT)));
#if 0
	checkCudaErrors(cudaMalloc(&d_plan->fk,maxbatchsize*ms*mt*mu*
		sizeof(CUCPX)));
#endif
#if 0
	cudaStream_t* streams =(cudaStream_t*) malloc(d_plan->opts.gpu_nstreams*
		sizeof(cudaStream_t));
	for(int i=0; i<d_plan->opts.gpu_nstreams; i++)
		checkCudaErrors(cudaStreamCreate(&streams[i]));
	d_plan->streams = streams;
#endif
	return 0;
}

int allocgpumem3d_nupts(cufinufft_plan *d_plan)
/* 
	wrapper for gpu memory allocation in "setNUpts" stage.

	Melody Shih 07/25/19
*/
{
	int M = d_plan->M;
	int maxbatchsize = d_plan->maxbatchsize;

	d_plan->byte_now=0;
	// No extra memory is needed in nuptsdriven method;
	switch(d_plan->opts.gpu_method)
	{
		case 1:
			{
				if(d_plan->opts.gpu_sort)
					checkCudaErrors(cudaMalloc(&d_plan->sortidx, M*sizeof(int)));
				checkCudaErrors(cudaMalloc(&d_plan->idxnupts,M*sizeof(int)));
			}
			break;
		case 2:
			{
				checkCudaErrors(cudaMalloc(&d_plan->idxnupts,M*sizeof(int)));
				checkCudaErrors(cudaMalloc(&d_plan->sortidx, M*sizeof(int)));
			}
			break;
		case 4:
			{
				checkCudaErrors(cudaMalloc(&d_plan->sortidx,M*sizeof(int)));
			}
			break;
		default:
			cerr << "err: invalid method" << endl;
	}
#if 0
	checkCudaErrors(cudaMalloc(&d_plan->kx,M*sizeof(FLT)));
	checkCudaErrors(cudaMalloc(&d_plan->ky,M*sizeof(FLT)));
	checkCudaErrors(cudaMalloc(&d_plan->kz,M*sizeof(FLT)));
#endif
	checkCudaErrors(cudaMalloc(&d_plan->c,maxbatchsize*M*sizeof(CUCPX)));

	return 0;
}
void freegpumemory3d(cufinufft_plan *d_plan) 
/* 
	wrapper for freeing gpu memory.

	Melody Shih 07/25/19
*/
{
	cudaFree(d_plan->fw);
	//cudaFree(d_plan->kx);
	//cudaFree(d_plan->ky);
	//cudaFree(d_plan->kz);
	//cudaFree(d_plan->c);
	cudaFree(d_plan->fwkerhalf1);
	cudaFree(d_plan->fwkerhalf2);
	cudaFree(d_plan->fwkerhalf3);
	switch(d_plan->opts.gpu_method)
	{
		case 1:
			{
				if(d_plan->opts.gpu_sort){
					checkCudaErrors(cudaFree(d_plan->idxnupts));
					checkCudaErrors(cudaFree(d_plan->sortidx));
					checkCudaErrors(cudaFree(d_plan->binsize));
					checkCudaErrors(cudaFree(d_plan->binstartpts));
				}else{
					checkCudaErrors(cudaFree(d_plan->idxnupts));
				}
			}
			break;
		case 2:
			{
				checkCudaErrors(cudaFree(d_plan->idxnupts));
				checkCudaErrors(cudaFree(d_plan->sortidx));
				checkCudaErrors(cudaFree(d_plan->numsubprob));
				checkCudaErrors(cudaFree(d_plan->binsize));
				checkCudaErrors(cudaFree(d_plan->binstartpts));
				checkCudaErrors(cudaFree(d_plan->subprobstartpts));
				checkCudaErrors(cudaFree(d_plan->subprob_to_bin));
			}
			break;
		case 4:
			{
				checkCudaErrors(cudaFree(d_plan->idxnupts));
				checkCudaErrors(cudaFree(d_plan->sortidx));
				checkCudaErrors(cudaFree(d_plan->numsubprob));
				checkCudaErrors(cudaFree(d_plan->binsize));
				checkCudaErrors(cudaFree(d_plan->binstartpts));
				checkCudaErrors(cudaFree(d_plan->subprobstartpts));
				checkCudaErrors(cudaFree(d_plan->subprob_to_bin));
			}
			break;
	}
	for(int i=0; i<d_plan->opts.gpu_nstreams; i++)
		checkCudaErrors(cudaStreamDestroy(d_plan->streams[i]));
}
