#pragma once

#include <numeric>
#include <algorithm>
#include <cub/util_macro.cuh>

const int NGPUS = 1;

const int N = 96;
const int RakeWidth = 8;

const int Times = 4;

const int ILP = 5;    // 96 維，不能設 8 ，因為 P = 97, Pitch = 104, 104-97=7 => 最後只有 7 個給你用
const int InstLP = 1; // 改的話會影響 sort 過的 reduce

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

   float norm() const { return std::inner_product(data, data + P, data, 0); }
   void minimize()
   {
      auto p = std::min_element(data, data + P, [](float a, float b) { return a * a < b * b; });
      std::rotate(data, p, data + P);
   }

   template <class Archive>
   void serialize(Archive & ar, const unsigned int version)
   {
      ar & data;
   }
};

struct NotReduced
{
   __host__ __device__ bool operator()(Norm n) const { return n > 1000000; }
};

struct NotCollision
{
   __host__ __device__ bool operator()(Norm n) const { return n > 1; }
};

template <int step>
__global__
void reduce(Point* gs, Norm* gns, size_t g_size, const Point* hs, const Norm* hns, size_t h_size);

__global__ void minimize(Point* list, size_t size);
