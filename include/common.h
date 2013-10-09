#ifndef _COMMON_H_
#define _COMMON_H_

#ifndef M_PI
#define M_PI (3.1415926535897932384626)
#endif

#define _2M_PI (2*M_PI)

#define PROBLEM_SIZE 256

//Template function collections...
template<typename Float>
void create_field(Float *out, const int nx, const int ny, const int nz,
                  const Float dx, const Float dy, const Float dz,
                  const Float kappa, const Float time) 
{
  Float damping   = (_2M_PI)*(_2M_PI)*kappa;
  Float exponent  = exp(-damping*time);
  for (int k = 0; k < nz; k++) 
    for (int j = 0; j < ny; j++) 
      for (int i = 0; i < nx; i++) 
      {
        int s = k*nx*ny + j*nx + i;
        Float x = dx*((Float)(i + 0.5));
        Float y = dy*((Float)(j + 0.5));
        Float z = dz*((Float)(k + 0.5));
        out[s] = (Float)0.125
          *(1.0 - exponent*cos(_2M_PI*x))
          *(1.0 - exponent*cos(_2M_PI*y))
          *(1.0 - exponent*cos(_2M_PI*z));
      }
   return;  
}

template<typename Float>
Float accuracy(const Float *in1, Float *in2, const int len) {
  double err   = 0.0;
  double norm  = 0.0;  
  for (int i = 0; i < len; i++)
  {  
    err  += (in1[i] - in2[i])*(in1[i] - in2[i]);
    norm += in1[i]*in1[i];
  }
  return (Float)sqrt(err/norm);
}


#endif
