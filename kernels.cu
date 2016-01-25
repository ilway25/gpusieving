#include <cub/cub.cuh>

#include "kernels.cuh"

template __global__ void reduce<0>(Point*, Norm*, size_t, const Point*, const Norm*, size_t);
template __global__ void reduce<1>(Point*, Norm*, size_t, const Point*, const Norm*, size_t);
template __global__ void reduce<2>(Point*, Norm*, size_t, const Point*, const Norm*, size_t);

// __device__ float tmp;

const int NumPrefetch = CUB_QUOTIENT_FLOOR(4 * BlockDim, Pitch);

using shared_t = union
{
   float block[BlockDim][4];
   float linear[NumPrefetch][Pitch];
};

template <int step>
__global__
void reduce(Point* gs, Norm* gns, size_t g_size, const Point* hs, const Norm* hns, size_t h_size)
{
   __shared__ bool check;
   if (step == 0)
   {
      if (threadIdx.x == 0) check = false;
      __syncthreads();
   }

   const int subidx  = threadIdx.x % RakeWidth;
   const int subinst = threadIdx.x / RakeWidth;

   float* g_ptr = reinterpret_cast<float*>(gs);
   const float* h_ptr = reinterpret_cast<const float*>(hs);

   cub::CacheModifiedInputIterator<cub::LOAD_LDG, float> g_in(g_ptr);
   cub::CacheModifiedInputIterator<cub::LOAD_LDG, float> h_in(h_ptr);

   using BlockLoadT = cub::BlockLoad<decltype(g_in), BlockDim, NT, cub::BLOCK_LOAD_VECTORIZE>;
   using BlockStoreT = cub::BlockStore<float*, BlockDim, NT, cub::BLOCK_STORE_VECTORIZE>;
   using BlockLoadVT = cub::BlockLoad<decltype(h_in), BlockDim, 4, cub::BLOCK_LOAD_VECTORIZE>;

   union {
      typename BlockLoadT::TempStorage load;
      typename BlockStoreT::TempStorage store;
      typename BlockLoadVT::TempStorage loadv;
   } shared;


   for (int g_base = blockIdx.x * InstPerBlock; g_base < g_size; g_base += GridDim * InstPerBlock)
   {
      const int g_idx = g_base + subinst;

      float g[NT], gg;
      float reduced {}; // Flag: 0 <-> Not reduced.

      BlockLoadT(shared.load).Load(g_in + g_base * Pitch, g);
      gg = gns[g_idx];

      float min_norm = gg + P * (g_in + g_base * Pitch)[0] * (g_in + g_base * Pitch)[0];

      __shared__ alignas(128) shared_t prefetch;
      __shared__ float prefetch_n[BlockDim];

      for (int h_base = 0; h_base < h_size; h_base += NumPrefetch)
      {
         BlockLoadVT(shared.loadv).Load(h_in + h_base * Pitch, prefetch.block[threadIdx.x]);
         if (threadIdx.x < NumPrefetch)
            prefetch_n[threadIdx.x] = hns[h_base + threadIdx.x];
         __syncthreads();

         if (step == 0)
         {
            if (__all(gg < prefetch_n[0]) && threadIdx.x == 0)
               check = true;
            __syncthreads();

            if (check) break;
         }

         for (int i = 0; i < NumPrefetch && h_base + i < h_size; ++i)
         {
            __syncthreads();

            const int h_idx = h_base + i;
            const float hh = prefetch_n[i];

            if (hh < 10) continue; // h is already reduced

            // h_buf has no zero padding
            using sep = float[RakeWidth][NT];
            __shared__ float h_buf[BlockDim];

            if (threadIdx.x < P)
            {
               h_buf[threadIdx.x] = prefetch.linear[i][threadIdx.x];
               h_buf[threadIdx.x + P] = 0;
            }
            if (threadIdx.x < Pitch * 2 - P * 2)
               h_buf[threadIdx.x + 2 * P] = 0;


            for (int rot = 0; rot < CUB_ROUND_DOWN_NEAREST(P, 4) - 4; rot += 4)
            {
               __syncthreads();

               float q_best {};
               float gh {}, gh2 {}, gh3 {}, gh4 {};

               float h[NT + 3];
               for (int j = 0; j < NT; ++j)
                  h[j] = (*(volatile sep*)(&h_buf[rot]))[subidx][j];
               h[NT] = h_buf[rot + (subidx + 1) * NT];
               h[NT + 1] = h_buf[rot + (subidx + 1) * NT + 1];
               h[NT + 2] = h_buf[rot + (subidx + 1) * NT + 2]; // 小心出界
               if (subidx == RakeWidth - 1) h[NT] = h[NT + 1] = h[NT + 2] = 0;

               for (int j = 0; j < NT; ++j)
               {
                  gh += g[j] * h[j];
                  gh2 += g[j] * h[j+1];
                  gh3 += g[j] * h[j+2];
                  gh4 += g[j] * h[j+3];
               }
               if (subidx == RakeWidth - 1)
               {
                  gh2 += g[NT - Padding - 1] * h_buf[rot];

                  gh3 += g[NT - Padding - 2] * h_buf[rot];
                  gh3 += g[NT - Padding - 1] * h_buf[rot + 1];

                  gh4 += g[NT - Padding - 3] * h_buf[rot];
                  gh4 += g[NT - Padding - 2] * h_buf[rot + 1];
                  gh4 += g[NT - Padding - 1] * h_buf[rot + 2];
               }

               for (int j = 1; j < RakeWidth; j *= 2)
               {
                  gh += __shfl_xor(gh, j);
                  gh2 += __shfl_xor(gh2, j);
                  gh3 += __shfl_xor(gh3, j);
                  gh4 += __shfl_xor(gh4, j);
               }

               // if (threadIdx.x == 0 && blockIdx.x == 0)
               //    tmp = gh+gh2;

               int from = 0;
               for (int j = 0; j < 2; ++j) // j 可以 1 至 NT
               {
                  float uu = gg + (P * g[j]) * g[j],
                        uv = gh + (P * g[j]) * h[j],
                        vv = hh + P * h[j] * h[j];
                  float uv2 = gh2 + (P * g[j]) * h[j + 1],
                        vv2 = hh + P * h[j + 1] * h[j + 1];
                  float uv3 = gh3 + (P * g[j]) * h[j + 2],
                        vv3 = hh + P * h[j + 2] * h[j + 2];
                  float uv4 = gh4 + (P * g[j]) * h[j + 3],
                        vv4 = hh + P * h[j + 3] * h[j + 3];

                  float q = rintf(uv / uu);
                  float q2 = rintf(uv2 / uu);
                  float q3 = rintf(uv3 / uu);
                  float q4 = rintf(uv4 / uu);

                  if (step == 1 && gg < 0) q = q2 = q3 = q4 = 0;
                  if (step == 1 && g_idx == h_idx && rot == 0) q = 0;

                  float new_norm = uu + q * (q * vv - 2 * uv);
                  float new_norm2 = uu + q2 * (q2 * vv2 - 2 * uv2);
                  float new_norm3 = uu + q3 * (q3 * vv3 - 2 * uv3);
                  float new_norm4 = uu + q4 * (q4 * vv4 - 2 * uv4);

                  if (new_norm < min_norm) // 若 j 快到 NT，要加 && subidx * NT + j < P)
                  {
                     min_norm = new_norm;
                     q_best = q;
                     from = 0;
                  }

                  if (new_norm2 < min_norm) // 若 j 快到 NT，要加 && subidx * NT + j < P)
                  {
                     min_norm = new_norm2;
                     q_best = q2;
                     from = 1;
                  }

                  if (new_norm3 < min_norm) // 若 j 快到 NT，要加 && subidx * NT + j < P)
                  {
                     min_norm = new_norm3;
                     q_best = q3;
                     from = 2;
                  }

                  if (new_norm4 < min_norm) // 若 j 快到 NT，要加 && subidx * NT + j < P)
                  {
                     min_norm = new_norm4;
                     q_best = q4;
                     from = 3;
                  }
               }

               for (int j = 1; j < RakeWidth; j *= 2)
               {
                  float min_norm_t = __shfl_xor(min_norm, j);
                  float q_best_t = __shfl_xor(q_best, j);
                  float from_t = __shfl_xor(from, j);

                  if (min_norm_t < min_norm || min_norm_t == min_norm && (subidx ^ j) >= subidx)
                  {
                     min_norm = min_norm_t;
                     q_best = q_best_t;
                     from = from_t;
                  }
               }

               if (step == 0 || __any(q_best != 0)) // 這行舊版沒有
               {
                  // if (!(step == 1 && gg < 10))
                  {

                     if (subidx == RakeWidth - 1)
                     {
                        if (from >= 1)
                           h[NT - Padding] = h_buf[rot];
                        if (from >= 2)
                           h[NT - Padding + 1] = h_buf[rot + 1];
                        if (from >= 3)
                           h[NT - Padding + 2] = h_buf[rot + 2];
                     }

                     if (from == 0)
                        for (int j = 0; j < NT; ++j)
                           g[j] -= q_best * h[j];
                     else if (from == 1)
                        for (int j = 0; j < NT; ++j)
                           g[j] -= q_best * h[j + 1];
                     else if (from == 2)
                        for (int j = 0; j < NT; ++j)
                           g[j] -= q_best * h[j + 2];
                     else if (from == 3)
                        for (int j = 0; j < NT; ++j)
                           g[j] -= q_best * h[j + 3];

                     if (from == 0)
                        gg += q_best * (q_best * hh - 2 * gh);
                     else if (from == 1)
                        gg += q_best * (q_best * hh - 2 * gh2);
                     else if (from == 2)
                        gg += q_best * (q_best * hh - 2 * gh3);
                     else if (from == 3)
                        gg += q_best * (q_best * hh - 2 * gh4);

                     reduced += q_best * q_best;
                  }
               }

               __syncthreads();

               if (threadIdx.x == 0)
               {
                  h_buf[P + rot] = h[0];
                  h_buf[P + rot + 1] = h[1];
                  h_buf[P + rot + 2] = h[2];
                  h_buf[P + rot + 3] = h[3];
               }
            }
         }
         __syncthreads();
      }

      BlockStoreT(shared.store).Store(g_ptr + g_base * Pitch, g);
      if (reduced > 0.5) gns[g_idx] = -1;
      __syncthreads();

      if (step == 0 && check) break;
   }
}

struct DataWithTid
{
   __device__ DataWithTid() {}
   __device__ DataWithTid(float d, int t) : data(d), tid(t) {}

   float data;
   int tid;
};

struct DataWithTidMin
{
   __device__ DataWithTid operator() (const DataWithTid &a, const DataWithTid &b) const
   {
      if (a.data < b.data) return a;
      if (a.data > b.data) return b;
      if (a.tid < b.tid) return a;
      return b;
   }
};

__global__
void minimize(Point* list, size_t size)
{
   typedef cub::BlockReduce<DataWithTid, P> BlockReduce;
   __shared__ typename BlockReduce::TempStorage temp_storage;

   for (int i = blockIdx.x; i < size; i += GridDim)
   {
      float r = 1e20;
      if (threadIdx.x < P) r = list[i][threadIdx.x];

      DataWithTid diff(r * r, threadIdx.x);
      DataWithTid t = BlockReduce(temp_storage).Reduce(diff, DataWithTidMin());

      __shared__ DataWithTid min;
      if (threadIdx.x == 0) min = t;
      __syncthreads();

      int target = threadIdx.x - min.tid;
      if (target < 0) target += P;

      if (threadIdx.x < P)
         list[i][target] = r;
      __syncthreads();
   }
}

#if 0
__global__
void reduce(Point* gs, Norm* gns, size_t g_size, const Point* hs, const Norm* hns, size_t h_size)
{
   __shared__ bool check;
   if (step == 0)
   {
      if (threadIdx.x == 0) check = false;
      __syncthreads();
   }

   const int subidx  = threadIdx.x % RakeWidth;
   const int subinst = threadIdx.x / RakeWidth;

   float* g_ptr = reinterpret_cast<float*>(gs);
   const float* h_ptr = reinterpret_cast<const float*>(hs);

   cub::CacheModifiedInputIterator<cub::LOAD_LDG, float> g_in(g_ptr);
   cub::CacheModifiedInputIterator<cub::LOAD_LDG, float> h_in(h_ptr);

   using BlockLoadT = cub::BlockLoad<decltype(g_in), BlockDim, NT, cub::BLOCK_LOAD_VECTORIZE>;
   using BlockStoreT = cub::BlockStore<float*, BlockDim, NT, cub::BLOCK_STORE_VECTORIZE>;
   using BlockLoadVT = cub::BlockLoad<decltype(h_in), BlockDim, 4, cub::BLOCK_LOAD_VECTORIZE>;

   union {
      typename BlockLoadT::TempStorage load;
      typename BlockStoreT::TempStorage store;
      typename BlockLoadVT::TempStorage loadv;
   } shared;


   for (int g_base = blockIdx.x * InstPerBlock; g_base < g_size; g_base += GridDim * InstPerBlock)
   {
      const int g_idx = g_base + subinst;

      float g[NT], gg;
      float reduced {}; // Flag: 0 <-> Not reduced.

      BlockLoadT(shared.load).Load(g_in + g_base * Pitch, g);
      gg = gns[g_idx];

      float min_norm = gg + P * (g_in + g_base * Pitch)[0] * (g_in + g_base * Pitch)[0];

      __shared__ alignas(128) shared_t prefetch;
      __shared__ float prefetch_n[BlockDim];

      for (int h_base = 0; h_base < h_size; h_base += NumPrefetch)
      {
         BlockLoadVT(shared.loadv).Load(h_in + h_base * Pitch, prefetch.block[threadIdx.x]);
         if (threadIdx.x < NumPrefetch)
            prefetch_n[threadIdx.x] = hns[h_base + threadIdx.x];
         __syncthreads();

         if (step == 0)
         {
            if (__all(gg < prefetch_n[0]) && threadIdx.x == 0)
               check = true;
            __syncthreads();

            if (check) break;
         }

         for (int i = 0; i < NumPrefetch && h_base + i < h_size; ++i)
         {
            __syncthreads();

            const int h_idx = h_base + i;
            const float hh = prefetch_n[i];

            if (hh < 10) continue; // h is already reduced

            // h_buf has no zero padding
            using sep = float[RakeWidth][NT];
            __shared__ float h_buf[BlockDim];

            if (threadIdx.x < P)
            {
               h_buf[threadIdx.x] = prefetch.linear[i][threadIdx.x];
               h_buf[threadIdx.x + P] = 0;
            }
            if (threadIdx.x < Pitch * 2 - P * 2)
               h_buf[threadIdx.x + 2 * P] = 0;


            for (int rot = 0; rot < P; ++rot)
            {
               __syncthreads();

               float q_best {};
               float gh {};

               float h[NT];
               for (int j = 0; j < NT; ++j)
                  h[j] = (*(volatile sep*)(&h_buf[rot]))[subidx][j];

               for (int j = 0; j < NT; ++j)
                  gh += g[j] * h[j];

               for (int j = 1; j < RakeWidth; j *= 2)
                  gh += __shfl_xor(gh, j);

               for (int j = 0; j < 1; ++j) // j 可以 1 至 NT
               {
                  float uu = gg + (P * g[j]) * g[j],
                        uv = gh + (P * g[j]) * h[j],
                        vv = hh + P * h[j] * h[j];

                  float q = rintf(uv / uu);

                  // if (step == 1 && gg < 0) q = 0;
                  if (step == 1 && g_idx == h_idx && rot == 0) q = 0;

                  float new_norm = uu + q * (q * vv - 2 * uv);

                  // if (new_norm < min_norm && g_idx == 1648 && q != 0)
                     // printf(" -- %d %d %f\n", h_idx, rot, q);

                  if (new_norm < min_norm) // 若 j 快到 NT，要加 && subidx * NT + j < P)
                  {
                     min_norm = new_norm;
                     q_best = q;
                  }
               }

               for (int j = 1; j < RakeWidth; j *= 2)
               {
                  float min_norm_t = __shfl_xor(min_norm, j);
                  float q_best_t = __shfl_xor(q_best, j);

                  if (min_norm_t < min_norm || min_norm_t == min_norm && (subidx ^ j) >= subidx)
                  {
                     min_norm = min_norm_t;
                     q_best = q_best_t;
                  }
               }

               if (!(step == 1 && gg < 10))
               {
                  for (int j = 0; j < NT; ++j)
                     g[j] -= q_best * h[j];
                  gg += q_best * (q_best * hh - 2 * gh);
                  reduced += q_best * q_best;
               }

               __syncthreads();

               if (threadIdx.x == 0)
                  h_buf[P + rot] = h[0];
            }
         }
         __syncthreads();
      }

      BlockStoreT(shared.store).Store(g_ptr + g_base * Pitch, g);
      if (reduced > 0.5) gns[g_idx] = -1;
      __syncthreads();

      if (step == 0 && check) break;
   }
}
#endif
