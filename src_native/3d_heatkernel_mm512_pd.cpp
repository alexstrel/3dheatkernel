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

/*---------double precision 7pt stencile----------*/

#include "stencile_intrin.h"

void compute_7pt_stencile_mm512_pd
(double *out, const double *in, const int Lx, const int Ly, const int Lz, const double C1, const double C2)
{
#include "mm512pd_3d_hk_core.h"
  return;
}

int main(int argc, char *argv[]) 
{
#ifdef _OPENMP

#ifndef KMP_AFFINITY
  kmp_set_defaults("KMP_AFFINITY=compact, granularity=fine");
//  kmp_set_defaults("KMP_AFFINITY=scatter, granularity=fine");
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

  const int problem_size = sizeof(double)*nx*ny*nz; 

  double *f1 = (double *)_mm_malloc(problem_size, 64);
  double *f2 = (double *)_mm_malloc(problem_size, 64);

  assert(f1 != MAP_FAILED);
  assert(f2 != MAP_FAILED);

  double *answer = (double *)_mm_malloc(problem_size, 64);
  double *f_final = NULL;


  int    count = 0;  
  double c0, c1;

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

  c1 = kappa*dt/(dx*dx);
  c0 = 1.0 - 6*c1;

  printf("Running heat kernel equation %d times\n", count); 
  fflush(stdout);

  double *f1_t = f1;
  double *f2_t = f2;

  struct timeval time_begin, time_end;

  gettimeofday(&time_begin, NULL);

  for (int i = 0; i < count; ++i) {
    compute_7pt_stencile_mm512_pd(f2_t, f1_t, nx, ny, nz, c0, c1); 
    double *t = f1_t;
    f1_t    = f2_t;
    f2_t    = t;
  }
  gettimeofday(&time_end, NULL);

  double time = count * dt;

  create_field<double>(answer, nx, ny, nz, dx, dy, dz, kappa, time);

  double err = accuracy<double>(answer,f_final, nx*ny*nz);
  double elapsed_time = (time_end.tv_sec - time_begin.tv_sec)
      + (time_end.tv_usec - time_begin.tv_usec)*1.0e-6;
  double Gflops = (nx*ny*nz)*8.0*count/elapsed_time * 1.0e-09;
  double Gstens = (nx*ny*nz)*1.0*count/elapsed_time * 1.0e-06;
  double thput = (nx * ny * nz) * sizeof(double) * 3.0 * count
      / elapsed_time * 1.0e-09;

  fprintf(stderr, "Elapsed time : %.3f (s)\n", elapsed_time);
  fprintf(stderr, "FLOPS        : %.3f (GFlops)\n", Gflops);
  fprintf(stderr, "Updates      : %.3f (Mupdates/sec)\n", Gstens);
  fprintf(stderr, "Throughput   : %.3f (GB/s)\n", thput);  
  fprintf(stderr, "Accuracy     : %e\n", err);
  
  _mm_free(f1);
  _mm_free(f2);
  _mm_free(answer);
  return 0;
}
