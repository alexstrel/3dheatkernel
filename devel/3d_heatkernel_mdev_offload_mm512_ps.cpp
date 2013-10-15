#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>
#include <assert.h>
#include <sys/mman.h>
#include <offload.h>
#include <offload.h> 

#include "common.h"

const int problem_dim = PROBLEM_SIZE;

/*---------single precision 7pt stencile----------*/

#pragma offload_attribute(push,target(mic))

#include "stencile_intrin.h"

/*
template <>
void compute_7pt_stencile_mm512_ps
(float *out, const float *in, const int Lx, const int Ly, const int Lz, const float C0, const float C1)
{
//use this macro for the offload mode:
#ifdef __MIC__
#include "mm512ps_3d_hk_core.h"
#else
  printf("\nThe heat kernel stencile operator was not compiled for the host..\n");
  exit(-1);
#endif //__MIC__
  return;
}
*/

#define INTERN
void compute_intern_7pt_stencile_mm512_ps
(float *out, const float *in, const int Lx, const int Ly, const int Lz, const float C0, const float C1)
{
//use this macro for the offload mode:
#ifdef __MIC__
#include "mm512ps_3d_hk_core.h"
#else
  printf("\nThe heat kernel stencile operator was not compiled for the host..\n");
  exit(-1);
#endif //__MIC__
  return;
}

#undef INTERN

void compute_extern_7pt_stencile_mm512_ps
(float *out, const float *in, const int Lx, const int Ly, const int Lz, const float C0, const float C1)
{
//use this macro for the offload mode:
#ifdef __MIC__
#include "mm512ps_3d_hk_core.h"
#else
  printf("\nThe heat kernel stencile operator was not compiled for the host..\n");
  exit(-1);
#endif //__MIC__
  return;
}

#pragma offload_attribute(pop)

int main(int argc, char *argv[]) 
{
  printf("Checking for Intel(R) Xeon Phi(TM) (Target CPU) devices...\n\n");
#ifdef __INTEL_OFFLOAD
  const int num_dev = _Offload_number_of_devices();
#else
  const int num_dev = 0;
#endif
  printf("Number of Target devices installed: %d\n\n",num_devices);

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

  const int    lnz   = nz / num_dev;

  const int V  = nx*ny*nz ;
  const int LV = nx*ny*lnz;
  const int FV = nx*ny;//face volume
  const int GV = 2*FV;//ghost volume

  const int problem_size = sizeof(float)*V; 
  const int face_size   = sizeof(float)*nx*ny;//and we have two faces

  //allocate original (host) fields
  float *f1 = (float *)_mm_malloc(problem_size, 64);
  float *f2 = (float *)_mm_malloc(problem_size, 64);
  assert(!f1 || !f2);

  //device memory objects, including ghost zones (don't exist on the host)
  float *local_f1;
  float *local_f2;
 
  //float *fwd_ghost = (float *)_mm_malloc(ghost_size, 64);//get from (rank+1)
  //float *bwd_ghost = (float *)_mm_malloc(ghost_size, 64);//get from (rank-1)

  float *fwd_face  = (float *)_mm_malloc(num_dev*face_size, 64); //send to rank+1
  float *bwd_face  = (float *)_mm_malloc(num_dev*face_size, 64); //send to rank-1

  const int fwd_face_offset = (LV-nx*ny);

  float *answer  = (float *)_mm_malloc(problem_size, 64);
  float *f_final = 0;

  memset(fwd_face, 0, num_dev*face_size);
  memset(bwd_face, 0, num_dev*face_size);

  int   count = 0;  

  float c1, c2;

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
  memset(f2, 0, problem_size);

  c2 = kappa*dt/(dx*dx);
  c1 = 1.0 - 6.0*c2;

  printf("Running heat kernel %d times\n", count); 
  fflush(stdout);

  struct timeval offload_time_begin, offload_time_end;
  struct timeval compute_time_begin, compute_time_end;

//copy in (allocate and retain):
  gettimeofday(&offload_time_begin, NULL);

//looks strange...any idea how to make it cleaner??
  for(int i = 0; i < num_dev; i++)
  {
     const int device_volume = LV+GV;//local volume + ghost zone   
#pragma offload_transfer target(mic:i) nocopy(local_f1: length(device_volume) alloc_if(1) free_if(0) align(64))\
                                       nocopy(local_f2: length(device_volume) alloc_if(1) free_if(0) align(64)) //a bit contr-intuitive 
//now moving bulk data:
     const int bgn = i*LV;
     const int end = bgn + LV;
#pragma offload_transfer target(mic:i) in(f1[bgn:end]:into(local_f1[0:LV])) signal(&f1[bgn]) 
#pragma offload_transfer target(mic:i) in(f2[bgn:end]:into(local_f2[0:LV])) signal(&f2[bgn]) 
  }

  __attribute__ ((target(mic))) float _mc1 = c1;
  __attribute__ ((target(mic))) float _mc2 = c2;

  __attribute__ ((target(mic))) int _mnx = nx;
  __attribute__ ((target(mic))) int _mny = ny;
  __attribute__ ((target(mic))) int _mnz = lnz;//note:this is local size

  gettimeofday(&compute_time_begin, NULL);

  for (int i = 0; i < count/2; ++i) {
//***start the first phase***:
//first, exchange faces...
     for(int i = 0; i < num_dev; i++)
     {
//start communications here:
        const int bgn = i*FV;
        const int end = bgn+FV;
#pragma offload_transfer target(mic:i) signal(local_f1) out(local_f1[fwd_face_offset:fwd_face_offset+FV]:into(fwd_face[bgn:end])) \
                                                        out(local_f1[0:FV]:into(bwd_face[bgn:end])) 
     }
//start interior computations:
#define INTERIOR
     for(int i = 0; i < num_dev; i++)
     {
#pragma offload target(mic:i) signal(local_f2) nocopy(local_f2: length(LV) alloc_if(0) free_if(0) align(64)) \
                                               nocopy(local_f1: length(LV) alloc_if(0) free_if(0) align(64)) 
       compute_7pt_stencile_mm512_ps(local_f2, local_f1, _mnx, _mny, _mnz, _mc1, _mc2); 
     }
#undef  INTERIOR
     for(int i = 0; i < num_dev; i++)
     {
        const int bwd_bgn = i == 0 ? 0 : (i-1)*FV;
        const int bwd_end = bwd_bgn+FV
//
        const int fwd_bgn = i == (num_dev-1) ? i*FV : (i+1)*FV;
        const int fwd_end = fwd_bgn+FV :;
//this is really no so good...:(
//start exterior computations:
#pragma offload target(mic:i) wait(local_f1) nocopy(local_f2: length(LV) alloc_if(0) free_if(0) align(64)) \
                                             nocopy(local_f1: length(LV) alloc_if(0) free_if(0) align(64)) \
                                             in(bwd_face[bwd_bgn:bwd_end]:into(local_f1[LV:FV]))\
                                             in(fwd_face[fwd_bgn:fwd_end]:into(local_f1[LV+FV:LV+2*FV]))\
                                             signal(local_f2)
       compute_7pt_stencile_mm512_ps(local_f2, local_f1, _mnx, _mny, _mnz, _mc1, _mc2);
     }

//***start the second phase***:
/*
     for(int i = 0; i < num_dev; i++)
     {
//start communications here:
//first, exchange faces:
       const int end = bgn+FV;
#pragma offload_transfer target(mic:i) out(local_f1[fwd_face_offset:end]:into(fwd_face[0:FV])) signal(fwd_face)
#pragma offload_transfer target(mic:i) out(local_f1[0:FV]:into(bwd_face[0:FV])) signal(bwd_face)
     }
//start interior computations:
#define INTERIOR
     for(int i = 0; i < num_dev; i++)
     {
#pragma offload target(mic:i) nocopy(local_f2: length(LV) alloc_if(0) free_if(0) align(64)) \
                              nocopy(local_f1: length(LV) alloc_if(0) free_if(0) align(64)) \
                              signal(f1)
       compute_7pt_stencile_mm512_ps(local_f2, local_f1, _mnx, _mny, _mnz, _mc1, _mc2); 
     }
#undef  INTERIOR
     for(int i = 0; i < num_dev; i++)
     {
//this is really no so good...:(
#pragma offload_wait(fwd_face) target(mic:i)
#pragma offload_wait(bwd_face) target(mic:i)

//start exterior computations:
#pragma offload target(mic:i) nocopy(local_f2: length(LV) alloc_if(0) free_if(0) align(64)) \
                              nocopy(local_f1: length(LV) alloc_if(0) free_if(0) align(64)) \
                              in(fwd_face[:]:into(local_f1[??:??]))\
                              in(bwd_face[:]:into(local_f1[??:??]))\
                              signal(local_f1)
       compute_7pt_stencile_mm512_ps(local_f2, local_f1, _mnx, _mny, _mnz, _mc1, _mc2);
     }
*/
  }
//an extra term for odd count number:
  if(count & 1) {
 }

#pragma offload_wait target(mic:mic_id) wait(f1) 

  gettimeofday(&compute_time_end, NULL);
//transfer back to host:
//WARNING: uncommenting wait results in error: :offload error: device # does not have a pending signal for wait(address)"
#pragma offload_transfer target(mic:mic_id) out(f1, f2:length(V) free_if(1) align(64)) //wait(f1)
  gettimeofday(&offload_time_end, NULL);

  float time = count * dt;

  create_field<float>(answer, nx, ny, nz, dx, dy, dz, kappa, time);
  float err = accuracy<float>(answer,f_final, nx*ny*nz);

  double offload_elapsed_time = (offload_time_end.tv_sec - offload_time_begin.tv_sec)
      + (offload_time_end.tv_usec - offload_time_begin.tv_usec)*1.0e-6;
  float gflops = V*8.0*count/offload_elapsed_time * 1.0e-09;
  double thput = V*sizeof(float) * 3.0 * count
      / offload_elapsed_time * 1.0e-09;

  fprintf(stderr, "Offload time         : %.3f (s)\n", offload_elapsed_time);
  fprintf(stderr, "Effective FLOPS      : %.3f (GFlops)\n", gflops);
  fprintf(stderr, "Offload throughput   : %.3f (GB/s)\n", thput);

  double compute_elapsed_time = (compute_time_end.tv_sec - compute_time_begin.tv_sec)
      + (compute_time_end.tv_usec - compute_time_begin.tv_usec)*1.0e-6;
  gflops = V*8.0*count/compute_elapsed_time * 1.0e-09;
  float Gstens = (nx*ny*nz)*1.0*count/compute_elapsed_time * 1.0e-06;
  thput  = V*sizeof(float)*3.0*count / compute_elapsed_time * 1.0e-09;

  fprintf(stderr, "\nCompute time         : %.3f (s)\n", compute_elapsed_time);
  fprintf(stderr, "True FLOPS           : %.3f (GFlops)\n", gflops);
  fprintf(stderr, "Updates              : %.3f (Mupdates/sec)\n", Gstens);
  fprintf(stderr, "Device throughput    : %.3f (GB/s)\n", thput);

  fprintf(stderr, "\n\nAccuracy: %e\n", err);
  
  _mm_free(f1);
  _mm_free(f2);

  _mm_free(fwd_ghost);
  _mm_free(bwd_ghost);

  _mm_free(answer);
//shutdown MPI tasks:
  return 0;
}
