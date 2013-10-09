CXX = icpc
CC  = icc

MICSPCCOPT  = -mmic -std=c99 -O2 -vec-report6 -Wall -openmp -fimf-domain-exclusion=15 -fimf-precision=low
MICDPCCOPT  = -mmic -std=c99 -O2 -vec-report6 -Wall -openmp # -fimf-domain-exclusion=15

MICLDFLAG = -mmic -lm -openmp #-L/home/astrel/Install -lc

OFFCCOPT  = -std=c99 -O2 -vec-report6 -Wall -openmp -fimf-domain-exclusion=15 -fimf-precision=low
OFFLDFLAG = -lm  -openmp #-L/home/astrel/Install -lc

INC = -I. -Iinclude -Iinclude/3dheatkernel_core

all: 3d_heatkernel_mm512_ps.mic 3d_heatkernel_mm512_pd.mic 3d_heatkernel_offload_mm512_ps.off 3d_heatkernel_offload_mm512_pd.off

3d_heatkernel_mm512_ps.mic: ./obj/3d_heatkernel_mm512_ps.o
	$(CXX) -o  $@  $^  $(MICLDFLAG)   

./obj/3d_heatkernel_mm512_ps.o : ./src_native/3d_heatkernel_mm512_ps.cpp
	$(CXX) $(MICSPCCOPT) -c -o  $@  $?  $(INC)  

3d_heatkernel_mm512_pd.mic: ./obj/3d_heatkernel_mm512_pd.o
	$(CXX) -o  $@  $^  $(MICLDFLAG)   

./obj/3d_heatkernel_mm512_pd.o : ./src_native/3d_heatkernel_mm512_pd.cpp
	$(CXX) $(MICDPCCOPT) -c -o  $@  $?  $(INC) 

3d_heatkernel_offload_mm512_ps.off: ./obj/3d_heatkernel_offload_mm512_ps.o
	$(CXX) -o  $@  $^  $(OFFLDFLAG)   

./obj/3d_heatkernel_offload_mm512_ps.o : ./src_offload/3d_heatkernel_offload_mm512_ps.cpp
	$(CXX) $(OFFCCOPT) -c -o  $@  $?  $(INC)  

3d_heatkernel_offload_mm512_pd.off: ./obj/3d_heatkernel_offload_mm512_pd.o
	$(CXX) -o  $@  $^  $(OFFLDFLAG)   

./obj/3d_heatkernel_offload_mm512_pd.o : ./src_offload/3d_heatkernel_offload_mm512_pd.cpp
	$(CXX) $(OFFCCOPT) -c -o  $@  $?  $(INC) 

copy:
	cp *.mic *.off ./exe

clean:
	rm ./obj/*.o *.mic *.off	

.PHONY:	clean	

