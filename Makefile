NVCC    = nvcc
CC      = clang++-3.6
override CPPFLAG += --std=c++11 -g
LIBS    = -larmadillo
# LIBS    = -lntl -lgmp -lboost_serialization

# NVARCH = -arch=sm_52 -use_fast_math -I../cub-1.4.1  -lineinfo
# NVARCH  = -arch=sm_52 -use_fast_math -I../cub-1.5.1 -Icpp11-range -lineinfo
NVARCH  = -gencode arch=compute_35,code=sm_35 -gencode arch=compute_52,code=sm_52 -use_fast_math -I../cub-1.5.1 -lineinfo

# SRCS  = main.cu gsieve.cu
OBJS  = main.o gsieve.o kernels.o cub_wrapper.o

all: $(OBJS)
	$(NVCC) -o g $^ $(LIBS) $(CPPFLAG) $(NVARCH)

# sampler: sampler.o common.o
# 	$(CC) -o $@ $^ $(LIBS) $(CPPFLAG)

# kernels.o: kernels.cu kernels.cuh gsieve.cuh
# 	$(NVCC) -c $< $(CPPFLAG) $(NVARCH)

# cub_wrapper.o: cub_wrapper.cu cub_wrapper.cuh gsieve.cuh
# 	# /usr/local/cuda-7.0/bin/nvcc -c $< $(CPPFLAG) $(NVARCH)
# 	$(NVCC) -c $< $(CPPFLAG) $(NVARCH)

# gsieve.o: gsieve.cu gsieve.cuh kernels.cuh
# 	$(NVCC) -c $< $(CPPFLAG) $(NVARCH)

%.o: %.cc
	$(CC) -c $^ $(CPPFLAG) -I../cub-1.5.1

%.o: %.cu
	$(NVCC) -c $^ $(CPPFLAG) $(NVARCH)

t.cubin: kernels.cu
	$(NVCC) kernels.cu $(CPPFLAG) $(NVARCH) -cubin -o t.cubin -Xptxas=-v

s: t.cubin
	# cuobjdump -sass -fun `cuobjdump --dump-elf-symbols t.cubin | grep reduce | awk '{ print $$4 }'` t.cubin > t.s
	cuobjdump -sass -fun _Z6reduceP5PointPfmPKS_PKfm t.cubin > t.s
	sed -i '/^\s\{20,\}/d' t.s

ptx: kernels.cu
	$(NVCC) kernels.cu $(CPPFLAG) $(NVARCH) -ptx -o t.ptx -src-in-ptx

clean:
	rm -f *.o g

# clean2:
# 	rm -f gsieve_main.o gsieve.o common.o kernels.o g
