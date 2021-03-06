#ifndef MM512PS_3D_HK_MDEV_CORE_H_
#define MM512PS_3D_HK_MDEV_CORE_H_

#define MM512_PS_UNROLL 16
#define MM512_UNROLL MM512_PS_UNROLL

  const int LyLx = Lx * Ly;

  const int Lxm64       = Lx - MM512_UNROLL;
  const int Lym1        = Ly - 1;
  const int Lzm1        = Lz - 1;
  const int LyLxmLx     = LyLx - Lx;
  const int LzLyLxmLyLx = Lz*LyLx - LyLx;

  //__m512 _c0 __attribute__((aligned(64))) = _mm512_set1_ps(C0);
  //__m512 register c0 = _mm512_load_ps((float *)&_c0);

  //__m512 _c1 __attribute__((aligned(64))) = _mm512_set1_ps(C1);
  //__m512 register c1 = _mm512_load_ps((float *)&_c1);
 
 //or simply:
 __m512 register c0 = _mm512_set1_ps(C0);
 __m512 register c1 = _mm512_set1_ps(C1);

#ifndef INTERIOR
  const int bulkVolume = Lz*LyLx
  const int faceVolume = LyLx;
  const int zbgn = 0;
  const int zend = 2;
#else
  const int zbgn = 1;
  const int zend = Lzm1;
#endif

#pragma omp parallel
{
#pragma noprefetch out
#pragma omp for collapse(2)
  for(int z = zbgn; z < zend; z++)//note: z is a local coordinate to the subdomain... 
  for(int y = 0; y < Ly; y++)
  {
    /*z, y coordinates:*/
    int s  = y * Lx + z * LyLx;

    /*Load hopping coords w.r.t von Neumann BC:*/
    int s_yp1 = (y == Lym1) ? s : s + Lx;
    int s_ym1 = (y == 0)    ? s : s - Lx;

#ifdef INTERIOR
    int s_zp1 = s + LyLx;
    int s_zm1 = s - LyLx;

    if(z != (Lzm1 - 1)) _mm_prefetch((const char*)&in[s_zp1+LyLx], _MM_HINT_T1);
#else
    int s_zp1 = z ? s + (bulkVolume + faceVolume) : s + LyLx;
    int s_zm1 = z ? s - LyLx : s + bulkVolume;
#endif

    __m512 USE_REG_HINT o, i, ixp1, ixm1, iyp1, iym1, izp1, izm1;

    i    = _mm512_load_ps((float *)&in[s]);
    /*Y neighbours:*/
    iyp1 = _mm512_load_ps((float *)&in[s_yp1]);
    iym1 = _mm512_load_ps((float *)&in[s_ym1]);
    /* or: _mm512_extload_ps(void const* mt, _MM_UPCONV_PS_ENUM conv, _MM_BROADCAST32_ENUM bc, int hint);      */

    /*Z neighbours:*/
    izp1 = _mm512_load_ps((float *)&in[s_zp1]);
    izm1 = _mm512_load_ps((float *)&in[s_zm1]);

    ixp1 = _mm512_setzero_ps(); 
    ixm1 = _mm512_setzero_ps();

    /*X- neighbors:*/
    _mm512_fwd_dirichlet_shift_ps(ixm1, i, &in[s]);

    /*X+ neighbours:*/
    _mm512_unaligned_load_ps(ixp1, &in[s+1]);

    _mm512_stencile_ps(o, i, ixp1, ixm1, iyp1, iym1, izp1, izm1, c0, c1);
    /*Load back to main memory:*/
    _mm512_extstore_ps((float*)&out[s], o, _MM_DOWNCONV_PS_NONE, _MM_HINT_NT);
#pragma novector
    for(int x = MM512_UNROLL; x < Lxm64; x += MM512_UNROLL)
    {
      s_yp1 += MM512_UNROLL;
      s_ym1 += MM512_UNROLL;
      s_zp1 += MM512_UNROLL;
      s_zm1 += MM512_UNROLL;
      s     += MM512_UNROLL;

      i    = _mm512_load_ps((float *)&in[s]);
      iyp1 = _mm512_load_ps((float *)&in[s_yp1]);
      iym1 = _mm512_load_ps((float *)&in[s_ym1]);
      izp1 = _mm512_load_ps((float *)&in[s_zp1]);
      izm1 = _mm512_load_ps((float *)&in[s_zm1]);

      _mm512_unaligned_load_ps(ixm1, &in[s-1]);     
      _mm512_unaligned_load_ps(ixp1, &in[s+1]);

      _mm512_stencile_ps(o, i, ixp1, ixm1, iyp1, iym1, izp1, izm1, c0, c1);
      _mm512_extstore_ps((float*)&out[s], o, _MM_DOWNCONV_PS_NONE, _MM_HINT_NT);
    }
    s_yp1 += MM512_UNROLL;
    s_ym1 += MM512_UNROLL;
    s_zp1 += MM512_UNROLL;
    s_zm1 += MM512_UNROLL;
    s     += MM512_UNROLL;

    i    = _mm512_load_ps((float *)&in[s]); 
    iyp1 = _mm512_load_ps((float *)&in[s_yp1]);
    iym1 = _mm512_load_ps((float *)&in[s_ym1]);
    izp1 = _mm512_load_ps((float *)&in[s_zp1]);
    izm1 = _mm512_load_ps((float *)&in[s_zm1]);

    _mm512_unaligned_load_ps(ixm1, &in[s-1]);
 
    _mm512_bwd_dirichlet_shift_ps(ixp1, i, &in[s]);

    _mm512_stencile_ps(o, i, ixp1, ixm1, iyp1, iym1, izp1, izm1, c0, c1);
    _mm512_extstore_ps((float*)&out[s], o, _MM_DOWNCONV_PS_NONE, _MM_HINT_NT);
  }
}/*end of openmp region */

#undef MM512_UNROLL
#undef MM512_PS_UNROLL

#endif
