g++ -O3 -mavx2 -mfma -fopenmp -lm -I../include -I../include/3dheatkernel_core 3d_heatkernel_mm256_ps.cpp -o 3d_heatkernel_mm256_ps.cpu
g++ -O3 -mavx2 -mfma -fopenmp -lm -I../include -I../include/3dheatkernel_core 3d_heatkernel_mm256_pd.cpp -o 3d_heatkernel_mm256_pd.cpu
