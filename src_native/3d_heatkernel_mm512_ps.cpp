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

/*---------single precision 7pt stencile----------*/

#include "stencile_intrin.h"

void compute_7pt_stencile_mm512_ps
(float *out, const float *in, const int Lx, const int Ly, const int Lz, const float C0, const float C1)
{
#include "3d_hk_core_ps.h"
  return;
}

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

  const int problem_size = sizeof(float)*nx*ny*nz; 

  float *f1 = (float *)_mm_malloc(problem_size, 64);
  float *f2 = (float *)_mm_malloc(problem_size, 64);

  assert(f1 != MAP_FAILED);
  assert(f2 != MAP_FAILED);

  float *answer = (float *)_mm_malloc(problem_size, 64);
  float *f_final = NULL;

  int   count = 0;  

  float c0, c1;

  float l = 1.0;
  float kappa = 0.1;
  float dx = l / nx;
  float dy = l / ny;
  float dz = l / nz;

  float dt    = 0.1*dx*dx / kappa;
  float scale = 0.1;
  count = scale / dt;
  f_final = (count % 2)? f2 : f1;

  create_field<float>(f1, nx, ny, nz, dx, dy, dz, kappa, 0.0);

  c1 = kappa*dt/(dx*dx);
  c0 = 1.0 - 6*c1;

  printf("Running heat kernel %d times\n", count); 
  fflush(stdout);

  float *f1_t = f1;
  float *f2_t = f2;

  struct timeval time_begin, time_end;

  gettimeofday(&time_begin, NULL);

  for (int i = 0; i < count; ++i) {
    compute_7pt_stencile_mm512_ps(f2_t, f1_t, nx, ny, nz, c0, c1); 
    float *t = f1_t;
    f1_t    = f2_t;
    f2_t    = t;
  }
  gettimeofday(&time_end, NULL);

  float time = count * dt;
 
  create_field<float>(answer, nx, ny, nz, dx, dy, dz, kappa, time);
  float err = accuracy<float>(answer,f_final, nx*ny*nz);

  double elapsed_time = (time_end.tv_sec - time_begin.tv_sec)
      + (time_end.tv_usec - time_begin.tv_usec)*1.0e-6;
  float Gflops = (nx*ny*nz)*8.0*count/elapsed_time * 1.0e-09;
  float Gstens = (nx*ny*nz)*1.0*count/elapsed_time * 1.0e-06;
  double thput = (nx * ny * nz) * sizeof(float) * 3.0 * count
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
