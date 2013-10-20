#ifndef MM256PD_3D_HK_CORE_H_
#define MM256PD_3D_HK_CORE_H_

#define MM256_PD_UNROLL 4
#define MM256_UNROLL MM256_PD_UNROLL

  const int LyLx = Lx * Ly;

  const int Lxm32       = Lx - MM256_UNROLL;
  const int Lym1        = Ly - 1;
  const int Lzm1        = Lz - 1;
  const int LyLxmLx     = LyLx - Lx;
  const int LzLyLxmLyLx = Lz*LyLx - LyLx;

  //__m256d register c0 = _mm256_broadcast_sd((double *)&C0);
  //__m256d register c1 = _mm256_broadcast_sd((double *)&C1);
 
  //or simply:
  __m256d register c0 = _mm256_set1_pd(C0);
  __m256d register c1 = _mm256_set1_pd(C1);

  int tid = 0;
  int nthreads = 1; 

#pragma omp parallel num_threads(4) private(tid, nthreads)
{
#ifdef _OPENMP  
  	tid        = omp_get_thread_num();
  	nthreads   = omp_get_num_threads();
#endif

  const int delta_y    = Ly / nthreads;

//#pragma omp for collapse(2)
  for(int z = 0; z < Lz; z++)
  for(int y = tid*delta_y; y < (tid+1)*delta_y; y++)
  //1. for(int y = tid; y < Ly; y += nthreads)
  {
    /*z, y coordinates:*/
    int s  = y * Lx + z * LyLx;

    /*Load hopping coords w.r.t von Neumann BC:*/
    int s_yp1 = (y == Lym1) ? s : s + Lx;
    int s_ym1 = (y == 0)    ? s : s - Lx;

    int s_zp1 = (z == Lzm1) ? s : s + LyLx;
    int s_zm1 = (z == 0)    ? s : s - LyLx;

    if(z != (Lzm1 - 1)) _mm_prefetch((const char*)&in[s_zp1+LyLx], _MM_HINT_T1);

    __m256d register o, i, ixp1, ixm1, iyp1, iym1, izp1, izm1;

    i    = _mm256_load_pd((double *)&in[s]);      
    //Y neighbours:
    iyp1 = _mm256_load_pd((double *)&in[s_yp1]);      
    iym1 = _mm256_load_pd((double *)&in[s_ym1]);      
    //Z neighbours:
    izp1 = _mm256_load_pd((double *)&in[s_zp1]);      
    izm1 = _mm256_load_pd((double *)&in[s_zm1]);      
      
    //X+ neighbours:
    ixp1  = _mm256_loadu_pd((double *)&in[s + 1]);
    //X- neighbors:
#ifdef USE_AVX2
    _mm256_avx2_fwd_dirichlet_shift_pd(ixm1, &in[s]);
#else
#endif

#ifdef USE_AVX2
    _mm256_avx2_stencile_pd(o, i, ixp1, ixm1, iyp1, iym1, izp1, izm1, c0, c1);
#else
#endif
    //Load back to main memory:
    _mm256_stream_pd((double*)&out[s], o);
   
    for(int x = MM256_UNROLL; x < Lxm32; x += MM256_UNROLL)
    {
      s_yp1 += MM256_UNROLL; 
      s_ym1 += MM256_UNROLL;      
      s_zp1 += MM256_UNROLL;
      s_zm1 += MM256_UNROLL;
      s     += MM256_UNROLL;

      i    = _mm256_load_pd((double *)&in[s]);      
      iyp1 = _mm256_load_pd((double *)&in[s_yp1]);      
      iym1 = _mm256_load_pd((double *)&in[s_ym1]);      
      izp1 = _mm256_load_pd((double *)&in[s_zp1]);      
      izm1 = _mm256_load_pd((double *)&in[s_zm1]);      
      ixp1 = _mm256_loadu_pd((double *)&in[s + 1]);
      ixm1 = _mm256_loadu_pd((double *)&in[s - 1]);    
#ifdef USE_AVX2           
      _mm256_avx2_stencile_pd(o, i, ixp1, ixm1, iyp1, iym1, izp1, izm1, c0, c1);
#else
#endif
      //Load back to main memory:
      _mm256_stream_pd((double*)&out[s], o);
    }

    s_yp1 += MM256_UNROLL; 
    s_ym1 += MM256_UNROLL;      
    s_zp1 += MM256_UNROLL;
    s_zm1 += MM256_UNROLL;
    s     += MM256_UNROLL;

    i    = _mm256_load_pd((double *)&in[s]);      
    //Y neighbours:
    iyp1 = _mm256_load_pd((double *)&in[s_yp1]);      
    iym1 = _mm256_load_pd((double *)&in[s_ym1]);      
    //Z neighbours:
    izp1 = _mm256_load_pd((double *)&in[s_zp1]);      
    izm1 = _mm256_load_pd((double *)&in[s_zm1]);

    //X+ neighbours:
#ifdef USE_AVX2
    _mm256_avx2_bwd_dirichlet_shift_pd(ixp1, &in[s]);
#else
#endif
    //X- neighbors:
    ixm1 = _mm256_loadu_pd((double *)&in[s - 1]); 
#ifdef USE_AVX2   
    _mm256_avx2_stencile_pd(o, i, ixp1, ixm1, iyp1, iym1, izp1, izm1, c0, c1);
#else
#endif
    //Load back to main memory:
    _mm256_stream_pd((double*)&out[s], o);

  } 
}//end of openmp region 

#undef MM256_UNROLL
#undef MM256_PS_UNROLL

#endif
