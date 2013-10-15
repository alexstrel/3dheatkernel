#ifndef _STENCILE_INTRIN_H_
#define _STENCILE_INTRIN_H_

#include <immintrin.h>

#define USE_REG_HINT register

//Single precision intrinsics:
//expected latency : 5(mul+first add)+4*3(add)+5(madd)=22
#define _mm512_stencile_ps(o, i, ixp1, ixm1, iyp1, iym1, izp1, izm1, c1, c2) \
  o    = _mm512_mul_ps(c1, i);                                                       \
  ixm1 = _mm512_add_ps(ixp1, ixm1);                                                  \
  iym1 = _mm512_add_ps(iyp1, iym1);                                                  \
  izm1 = _mm512_add_ps(izp1, izm1);                                                  \
  iym1 = _mm512_add_ps(izm1, iym1);                                                  \
  ixm1 = _mm512_add_ps(iym1, ixm1);                                                  \
  o    = _mm512_fmadd_ps(ixm1, c2, o);                                               

//Register manipulations:
#define _mm512_fwd_cyclic_shift_ps(o, i)     \
  o  = _mm512_mask_swizzle_ps(_mm512_swizzle_ps(i, _MM_SWIZ_REG_CDAB), 0x5555, _mm512_swizzle_ps(i, _MM_SWIZ_REG_BADC), _MM_SWIZ_REG_CDAB);\
  o  = _mm512_mask_blend_ps(0x1111, o, _mm512_permute4f128_ps(o, _MM_PERM_CBAD));

#define _mm512_bwd_cyclic_shift_ps(o, i)                                                         \
  o  = _mm512_swizzle_ps(i, _MM_SWIZ_REG_DACB);                 \
  o  = _mm512_mask_swizzle_ps(o, 0xCCCC, o, _MM_SWIZ_REG_CDAB);                                \
  o  = _mm512_mask_blend_ps(0x8888, o, _mm512_permute4f128_ps(o, _MM_PERM_ADCB));

//old versions for dirichlet shifts:
#define _mm512_fwd_dirichlet_shift_ps_old(o, i)     \
  o  = _mm512_mask_swizzle_ps(_mm512_swizzle_ps(i, _MM_SWIZ_REG_CDAB), 0x5555, _mm512_swizzle_ps(i, _MM_SWIZ_REG_BADC), _MM_SWIZ_REG_CDAB);\
  o  = _mm512_mask_blend_ps(0x1111, o, _mm512_permute4f128_ps(o, _MM_PERM_CBAD));\
  o  = _mm512_mask_mov_ps(o, 0x0001, i);

#define _mm512_bwd_dirichlet_shift_ps_old(o, i)                                                         \
  o  = _mm512_mask_swizzle_ps(_mm512_swizzle_ps(i, _MM_SWIZ_REG_DACB), 0xCCCC, _mm512_swizzle_ps(i, _MM_SWIZ_REG_DACB), _MM_SWIZ_REG_CDAB);\
  o  = _mm512_mask_blend_ps(0x8888, o, _mm512_permute4f128_ps(o, _MM_PERM_ADCB));\
  o  = _mm512_mask_mov_ps(o, 0x8000, i);

//new versions for dirichlet shifts:
#define _mm512_fwd_dirichlet_shift_ps(o, i, i_ptr)     \
  o  = _mm512_loadunpackhi_ps(o, (char*)(i_ptr+15));   \
  o  = _mm512_mask_mov_ps(o, 0x0001, i);

#define _mm512_bwd_dirichlet_shift_ps(o, i, i_ptr)     \
  o  = _mm512_loadunpacklo_ps(o, (char*)(i_ptr+1));   \
  o  = _mm512_mask_mov_ps(o, 0x8000, i);

//unaligned load:
#define _mm512_unaligned_load_ps(o, ptr)	   	        \
  o   = _mm512_loadunpacklo_ps(o, (char*)ptr   );	\
  o   = _mm512_loadunpackhi_ps(o, (char*)ptr+64);


//Double precision intrinsics macro
#define _mm512_stencile_pd(o, i, ixp1, ixm1, iyp1, iym1, izp1, izm1, c1, c2) \
  o    = _mm512_mul_pd(c1, i);                                                       \
  ixm1 = _mm512_add_pd(ixp1, ixm1);                                                  \
  iym1 = _mm512_add_pd(iyp1, iym1);                                                  \
  izm1 = _mm512_add_pd(izp1, izm1);                                                  \
  iym1 = _mm512_add_pd(izm1, iym1);                                                  \
  ixm1 = _mm512_add_pd(iym1, ixm1);                                                  \
  o    = _mm512_fmadd_pd(ixm1, c2, o);

//'dirichlet' shifts:
#define _mm512_fwd_dirichlet_shift_pd(o, i, i_ptr)     \
  o  = _mm512_loadunpackhi_pd(o, (char*)(i_ptr+7));   \
  o  = _mm512_mask_mov_pd(o, 0x01, i);

#define _mm512_bwd_dirichlet_shift_pd(o, i, i_ptr)     \
  o  = _mm512_loadunpacklo_pd(o, (char*)(i_ptr+1));   \
  o  = _mm512_mask_mov_pd(o, 0x80, i);

//unaligned load:
#define _mm512_unaligned_load_pd(o, ptr)	   	        \
  o   = _mm512_loadunpacklo_pd(o, (char*)ptr   );	\
  o   = _mm512_loadunpackhi_pd(o, (char*)ptr+64);

#endif //_STENCILE_INTRIN_H_
