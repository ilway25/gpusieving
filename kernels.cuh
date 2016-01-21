#pragma once

#include <cub/util_macro.cuh>

const int NGPUS = 3;

const int N = 102;
const int RakeWidth = 8;

// const int ILP = 2;
const int GridDim  = 256;
const int BlockDim = 256;

const int NumSamples = 4096;

static_assert(BlockDim > N, "BlcokDim");

const int P       = N + 1;

const int Pitch   = CUB_ROUND_UP_NEAREST(P, CUB_ROUND_UP_NEAREST(RakeWidth, 4));
const int Padding = Pitch - P;
const int NT      = Pitch / RakeWidth;
const int InstPerBlock = BlockDim / RakeWidth;

using Norm = float;

struct Point
{
   float data[Pitch];

   __device__ __host__ float& operator[](size_t idx) { return data[idx]; }
   __device__ __host__ const float& operator[](size_t idx) const { return data[idx]; }
};

struct NotReduced
{
   __host__ __device__ bool operator()(Norm n) const { return n > 10; }
};

template <int step>
__global__
void reduce(Point* gs, Norm* gns, size_t g_size, const Point* hs, const Norm* hns, size_t h_size);