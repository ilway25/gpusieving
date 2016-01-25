NVCC    = nvcc
CC      = clang++-3.6
override CPPFLAG += --std=c++11
LIBS    = -larmadillo -lboost_serialization

# NVARCH = -arch=sm_52 -use_fast_math -I../cub-1.5.1 -lineinfo
NVARCH = -gencode arch=compute_35,code=sm_35 -gencode arch=compute_52,code=sm_52 -use_fast_math -I../cub-1.5.1 -lineinfo

OBJS  = main.o gsieve.o kernels.o cub_wrapper.o

all: $(OBJS)
	$(NVCC) -o g $^ $(LIBS) $(CPPFLAG) $(NVARCH)

# sampler: sampler.o common.o
# 	$(CC) -o $@ $^ $(LIBS) $(CPPFLAG)

%.o: %.cc
	$(CC) -c $^ $(CPPFLAG) -I../cub-1.5.1

%.o: %.cu
	$(NVCC) -c $^ $(CPPFLAG) $(NVARCH)

t.cubin: kernels.cu
	$(NVCC) kernels.cu $(CPPFLAG) -arch sm_52 -use_fast_math -I../cub-1.5.1 -lineinfo  -cubin -o t.cubin -Xptxas=-v

s: t.cubin
	cuobjdump -sass -fun `cuobjdump --dump-elf-symbols t.cubin | grep reduceILi0 | awk '{ print $$4 }'` t.cubin > t.s
	# cuobjdump -sass -fun _Z6reduceP5PointPfmPKS_PKfm t.cubin > t.s
	sed -i '/^\s\{20,\}/d' t.s

ptx: kernels.cu
	$(NVCC) kernels.cu $(CPPFLAG) -arch sm_52 -use_fast_math -I../cub-1.5.1 -lineinfo -ptx -o t.ptx -src-in-ptx

dot: t.cubin
	nvdisasm -c -fun 7 -cfg t.cubin | dot -o1.pdf -Tpdf

clean:
	rm -f *.o g

# clean2:
# 	rm -f gsieve_main.o gsieve.o common.o kernels.o g
