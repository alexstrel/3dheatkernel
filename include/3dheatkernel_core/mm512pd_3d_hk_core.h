#ifndef MM512PD_3D_HK_CORE_H_
#define MM512PD_3D_HK_CORE_H_

#define MM512_PD_UNROLL 8
#define MM512_UNROLL MM512_PD_UNROLL

  const int LyLx = Lx * Ly;

  const int Lxm64       = Lx - MM512_UNROLL;
  const int Lym1        = Ly - 1;
  const int Lzm1        = Lz - 1;
  const int LyLxmLx     = LyLx - Lx;
  const int LzLyLxmLyLx = Lz*LyLx - LyLx;


  __m512d register c1 = _mm512_set1_pd(C1);
  __m512d register c2 = _mm512_set1_pd(C2);

#pragma omp parallel
{
#pragma noprefetch out
#pragma omp for collapse(2)
  for(int z = 0; z < Lz; z++)
  for(int y = 0; y < Ly; y++)
  {
    /*z, y coordinates:*/
    int s  = y * Lx + z * LyLx;

    /*Load hopping coords w.r.t von Neumann BC:*/
    int s_yp1 = (y == Lym1) ? s : s + Lx;
    int s_ym1 = (y == 0)    ? s : s - Lx;

    int s_zp1 = (z == Lzm1) ? s : s + LyLx;
    int s_zm1 = (z == 0)    ? s : s - LyLx;

    if(z != (Lzm1 - 1)) _mm_prefetch((const char*)&in[s_zp1+LyLx], _MM_HINT_T1);

    __m512d USE_REG_HINT o, i, ixp1, ixm1, iyp1, iym1, izp1, izm1;

    i    = _mm512_load_pd((double *)&in[s]);
    /*Y neighbours:*/
    iyp1 = _mm512_load_pd((double *)&in[s_yp1]);
    iym1 = _mm512_load_pd((double *)&in[s_ym1]);
    /* or: _mm512_extload_pd(void const* mt, _MM_UPCONV_pd_ENUM conv, _MM_BROADCAST32_ENUM bc, int hint);      */

    /*Z neighbours:*/
    izp1 = _mm512_load_pd((double *)&in[s_zp1]);
    izm1 = _mm512_load_pd((double *)&in[s_zm1]);

    ixp1 = _mm512_setzero_pd();
    ixm1 = _mm512_setzero_pd();

    /*X- neighbors:*/
    _mm512_fwd_dirichlet_shift_pd(ixm1, i, &in[s]);

    /*X+ neighbours:*/
    _mm512_unaligned_load_pd(ixp1, &in[s+1]);

    _mm512_stencile_pd(o, i, ixp1, ixm1, iyp1, iym1, izp1, izm1, c1, c2);
    /*Load back to main memory:*/
    _mm512_extstore_pd((double*)&out[s], o, _MM_DOWNCONV_PD_NONE, _MM_HINT_NT);
#pragma novector
    for(int x = MM512_UNROLL; x < Lxm64; x += MM512_UNROLL)
    {
      s_yp1 += MM512_UNROLL;
      s_ym1 += MM512_UNROLL;
      s_zp1 += MM512_UNROLL;
      s_zm1 += MM512_UNROLL;
      s     += MM512_UNROLL;

      i    = _mm512_load_pd((double *)&in[s]);
      iyp1 = _mm512_load_pd((double *)&in[s_yp1]);
      iym1 = _mm512_load_pd((double *)&in[s_ym1]);
      izp1 = _mm512_load_pd((double *)&in[s_zp1]);
      izm1 = _mm512_load_pd((double *)&in[s_zm1]);

      _mm512_unaligned_load_pd(ixm1, &in[s-1]);     
      _mm512_unaligned_load_pd(ixp1, &in[s+1]);

      _mm512_stencile_pd(o, i, ixp1, ixm1, iyp1, iym1, izp1, izm1, c1, c2);
      _mm512_extstore_pd((double*)&out[s], o, _MM_DOWNCONV_PD_NONE, _MM_HINT_NT);
    }
    s_yp1 += MM512_UNROLL;
    s_ym1 += MM512_UNROLL;
    s_zp1 += MM512_UNROLL;
    s_zm1 += MM512_UNROLL;
    s     += MM512_UNROLL;

    i    = _mm512_load_pd((double *)&in[s]); 
    iyp1 = _mm512_load_pd((double *)&in[s_yp1]);
    iym1 = _mm512_load_pd((double *)&in[s_ym1]);
    izp1 = _mm512_load_pd((double *)&in[s_zp1]);
    izm1 = _mm512_load_pd((double *)&in[s_zm1]);

    _mm512_unaligned_load_pd(ixm1, &in[s-1]);
 
    _mm512_bwd_dirichlet_shift_pd(ixp1, i, &in[s]);

    _mm512_stencile_pd(o, i, ixp1, ixm1, iyp1, iym1, izp1, izm1, c1, c2);
    _mm512_extstore_pd((double*)&out[s], o, _MM_DOWNCONV_PD_NONE, _MM_HINT_NT);
  }
}/*end of openmp region */

#undef MM512_UNROLL
#undef MM512_PD_UNROLL

#endif
