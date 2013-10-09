#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>
#include <assert.h>
#include <sys/mman.h>

#include "common.h"

const int problem_dim = PROBLEM_SIZE;

const int mic_id = 0;

/*---------single precision 7pt stencile----------*/

#pragma offload_attribute(push,target(mic))

#include "stencile_intrin.h"

void compute_7pt_stencile_mm512_pd
(double *out, const double *in, const int Lx, const int Ly, const int Lz, const double C1, const double C2)
{
//use this macro for the offload mode:
#ifdef __MIC__
#include "3d_hk_core_pd.h"
#else
  printf("\nThe heat kernel stencile operator was not compiled for the host..\n");
  exit(-1);
#endif //__MIC__
  return;
}
#pragma offload_attribute(pop)

int main(int argc, char *argv[]) 
{
#ifdef _OPENMP

#ifndef KMP_AFFINITY
  kmp_set_defaults("KMP_AFFINITY=compact, granularity=fine");
//  kmp_set_defaults("KMP_AFFINITY=scatter, granularity=fine");//this gives much slower performance...
#endif

#pragma omp parallel
#pragma omp master

//#ifndef OMP_NUM_THREADS
 omp_set_num_threads(240);
//#endif

 printf("\nComputing 7-point stencil on Intel Xeon Phi in %d threads.\n", omp_get_num_threads());

#endif

  const int    nx    = problem_dim;
  const int    ny    = problem_dim;
  const int    nz    = problem_dim;

  const int V = nx*ny*nz;

  const int problem_size = sizeof(double)*V; 

  double *f1 = (double *)_mm_malloc(problem_size, 64);
  double *f2 = (double *)_mm_malloc(problem_size, 64);

  assert(f1 != MAP_FAILED);
  assert(f2 != MAP_FAILED);

  double *answer = (double *)_mm_malloc(problem_size, 64);
  double *f_final = 0;

  int   count = 0;  

  double c1, c2;

  double l = 1.0;
  double kappa = 0.1;
  double dx = l / nx;
  double dy = l / ny;
  double dz = l / nz;

  double dt    = 0.1*dx*dx / kappa;
  double scale = 0.1;
  count = scale / dt;
  f_final = (count % 2)? f2 : f1;

  create_field<double>(f1, nx, ny, nz, dx, dy, dz, kappa, 0.0);
  memset(f2, 0, problem_size);

  c2 = kappa*dt/(dx*dx);
  c1 = 1.0 - 6.0*c2;

  printf("Running heat kernel %d times\n", count); 
  fflush(stdout);

  struct timeval offload_time_begin, offload_time_end;
  struct timeval compute_time_begin, compute_time_end;

//copy in (allocate and retain):
  gettimeofday(&offload_time_begin, NULL);
#pragma offload_transfer target(mic:mic_id) in(f1: length(V) alloc_if(1) free_if(0) align(64))\
                                            nocopy (f2: length(V) alloc_if(1) free_if(0) align(64)) //a bit contr-intuitive but works...

  __attribute__ ((target(mic))) double _mc1 = c1;
  __attribute__ ((target(mic))) double _mc2 = c2;

  __attribute__ ((target(mic))) int _mnx = nx;
  __attribute__ ((target(mic))) int _mny = ny;
  __attribute__ ((target(mic))) int _mnz = nz;

  gettimeofday(&compute_time_begin, NULL);

  for (int i = 0; i < count/2; ++i) {
#pragma offload target(mic:mic_id) nocopy(f2: length(V) alloc_if(0) free_if(0) align(64)) \
                                   nocopy(f1: length(V) alloc_if(0) free_if(0) align(64)) \
                                   signal(f1)
     {
       compute_7pt_stencile_mm512_pd(f2, f1, _mnx, _mny, _mnz, _mc1, _mc2); 
       compute_7pt_stencile_mm512_pd(f1, f2, _mnx, _mny, _mnz, _mc1, _mc2);
     }
  }
//an extra term for odd count number:
  if(count & 1) {
#pragma offload target(mic:mic_id) nocopy(f2: length(V) alloc_if(0) free_if(0) align(64)) \
                                   nocopy(f1: length(V) alloc_if(0) free_if(0) align(64)) \
                                   signal(f1)
       compute_7pt_stencile_mm512_pd(f2, f1, nx, ny, nz, c1, c2); //transfer constants in this case
 }

#pragma offload_wait target(mic:mic_id) wait(f1) 

  gettimeofday(&compute_time_end, NULL);
//transfer back to host:
//WARNING: uncommenting wait results in error: :offload error: device # does not have a pending signal for wait(address)"
#pragma offload_transfer target(mic:mic_id) out(f1, f2:length(V) free_if(1) align(64)) //wait(f1)
  gettimeofday(&offload_time_end, NULL);

  double time = count * dt;

  create_field<double>(answer, nx, ny, nz, dx, dy, dz, kappa, time);
  double err = accuracy<double>(answer,f_final, nx*ny*nz);

  double offload_elapsed_time = (offload_time_end.tv_sec - offload_time_begin.tv_sec)
      + (offload_time_end.tv_usec - offload_time_begin.tv_usec)*1.0e-6;
  double gflops = V*8.0*count/offload_elapsed_time * 1.0e-09;
  double thput = V*sizeof(double) * 3.0 * count
      / offload_elapsed_time * 1.0e-09;

  fprintf(stderr, "Offload time         : %.3f (s)\n", offload_elapsed_time);
  fprintf(stderr, "Effective FLOPS      : %.3f (GFlops)\n", gflops);
  fprintf(stderr, "Offload throughput   : %.3f (GB/s)\n", thput);

  double compute_elapsed_time = (compute_time_end.tv_sec - compute_time_begin.tv_sec)
      + (compute_time_end.tv_usec - compute_time_begin.tv_usec)*1.0e-6;
  gflops = V*8.0*count/compute_elapsed_time * 1.0e-09;
  double Gstens = (nx*ny*nz)*1.0*count/compute_elapsed_time * 1.0e-06;
  thput  = V*sizeof(double)*3.0*count / compute_elapsed_time * 1.0e-09;

  fprintf(stderr, "\nCompute time         : %.3f (s)\n", compute_elapsed_time);
  fprintf(stderr, "True FLOPS           : %.3f (GFlops)\n", gflops);
  fprintf(stderr, "Updates              : %.3f (Mupdates/sec)\n", Gstens);
  fprintf(stderr, "Device throughput    : %.3f (GB/s)\n", thput);

  fprintf(stderr, "\n\nAccuracy: %e\n", err);
  
  _mm_free(f1);
  _mm_free(f2);
  _mm_free(answer);

  return 0;
}
