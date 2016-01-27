#include <cub/cub.cuh>

#include "kernels.cuh"

// Unroller
template <int N>
struct sequence {
    template <typename Lambda>
    static __forceinline__ __device__ void run(const Lambda& f) {
        sequence<N-1>::run(f);
        f(N-1);
    }
};

template <>
struct sequence<0> {
    template <typename Lambda>
    static __forceinline__ __device__ void run(const Lambda& f) {}
};

template __global__ void reduce<0>(Point*, Norm*, size_t, const Point*, const Norm*, size_t);
template __global__ void reduce<1>(Point*, Norm*, size_t, const Point*, const Norm*, size_t);
template __global__ void reduce<2>(Point*, Norm*, size_t, const Point*, const Norm*, size_t);

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


   for (int g_base = blockIdx.x * InstPerBlock * InstLP; g_base < g_size; g_base += GridDim * InstPerBlock * InstLP)
   {
      const auto g_idx = [&](int e) { return g_base + e * InstPerBlock + subinst; };

      float g[InstLP][NT], gg[InstLP];
      float reduced[InstLP] {}; // Flag: 0 <-> Not reduced.

      float min_norm[InstLP];

      sequence<InstLP>::run([&](int e)
      {
         BlockLoadT(shared.load).Load(g_in + (g_base + e * InstPerBlock) * Pitch, g[e]);
         gg[e] = gns[g_idx(e)];
         auto t = (g_in + (g_base + e * InstPerBlock) * Pitch)[0];
         min_norm[e] = gg[e] + P * t * t;
      });

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
            if (__all(gg[0] < prefetch_n[0]) && threadIdx.x == 0)
               check = true;
            __syncthreads();

            if (check) break;
         }

         for (int i = 0; i < NumPrefetch && h_base + i < h_size; ++i) // 可省？
         {
            __syncthreads();

            const int h_idx = h_base + i;
            const float hh = prefetch_n[i];

            if (step != 0 && hh < 10) continue; // h is already reduced

            // h_buf has no zero padding
            using sep = float[RakeWidth][NT];
            __shared__ float h_buf[BlockDim]; // 可能不夠 126 維？

            h_buf[threadIdx.x] = threadIdx.x < P ? prefetch.linear[i][threadIdx.x] : 0;

            for (int rot = 0; rot < CUB_ROUND_DOWN_NEAREST(P, ILP) - ILP; rot += ILP)
            {
               __syncthreads();

               float q_best[InstLP] {};
               float gh[InstLP][ILP] {};

               float h[NT + (ILP - 1)];
               for (int j = 0; j < NT; ++j)
                  h[j] = (*(volatile sep*)(&h_buf[rot]))[subidx][j];

               sequence<ILP - 1>::run([&](int k)
               {
                  h[NT + k] = h_buf[rot + (subidx + 1) * NT + k]; // 小心出界
                  // if (subidx == RakeWidth - 1) h[NT + k] = 0;
               });

               // sequence<InstLP>::run([&](int e) // 以下兩個簡化成下面那個
               // {
               //    sequence<ILP>::run([&](int k)
               //    {
               //       for (int j = 0; j < NT; ++j)
               //          gh[e][k] += g[e][j] * h[j + k];
               //    });
               // });

               // if (subidx == RakeWidth - 1)
               //    sequence<InstLP>::run([&](int e)
               //    {
               //       sequence<ILP>::run([&](int k)
               //       {
               //          for (int p = 0 ; p < k; ++p)
               //             gh[e][k] += g[e][NT - Padding - (k - p)] * h_buf[rot + p];
               //       });
               //    });

               sequence<ILP>::run([&](int k)
               {
                  sequence<InstLP>::run([&](int e)
                  {
                     for (int j = 0; j < NT; ++j)
                        gh[e][k] += g[e][j] * h[j + k];
                  });

                  if (subidx == RakeWidth - 1)
                     h[NT - Padding + k] = h_buf[rot + k];
               });

               for (int j = 1; j < RakeWidth; j *= 2)
                  sequence<InstLP>::run([&](int e)
                  {
                     sequence<ILP>::run([&](int k)
                     {
                        gh[e][k] += __shfl_xor(gh[e][k], j);
                     });
                  });

               int from[InstLP] {};
               for (int j = 0; j < Times; ++j) // j 可以 1 至 NT
               {
                  sequence<InstLP>::run([&](int e)
                  {
                     float uu = gg[e] + (P * g[e][j]) * g[e][j];

                     sequence<ILP>::run([&](int k)
                     {
                        float uv = gh[e][k] + (P * g[e][j]) * h[j + k],
                              vv = hh + P * h[j + k] * h[j + k];

                        float q = rintf(uv / uu);

                        if (step == 1 && gg[e] < 0) q = 0;
                        if (step == 1 && g_idx(e) == h_idx && rot == 0 && k == 0) q = 0;

                        float new_norm = uu + q * (q * vv - 2 * uv);
                        if (new_norm < min_norm[e]) // 若 j 快到 NT，要加 && subidx * NT + j < P)
                        {
                           min_norm[e] = new_norm;
                           q_best[e] = q;
                           from[e] = k;
                        }
                     });
                  });
               }

               float qq {};
               sequence<InstLP>::run([&](int e)
               {
                  for (int j = 1; j < RakeWidth; j *= 2)
                  {
                     float min_norm_t = __shfl_xor(min_norm[e], j);
                     float q_best_t = __shfl_xor(q_best[e], j);
                     float from_t = __shfl_xor(from[e], j);

                     if (min_norm_t < min_norm[e] || min_norm_t == min_norm[e] && (subidx ^ j) >= subidx)
                     {
                        min_norm[e] = min_norm_t;
                        q_best[e] = q_best_t;
                        from[e] = from_t;
                     }
                  }
                  qq += q_best[e] * q_best[e];
               });

               if (step == 0 || __any(qq != 0)) // 這行舊版沒有
               {
                  sequence<ILP>::run([&](int k)
                  {
                     k = ILP - 1 - k; // 反過來
                     sequence<InstLP>::run([&](int e)
                     {
                        if (from[e] == k)
                        {
                           gg[e] += q_best[e] * (q_best[e] * hh - 2 * gh[e][k]);
                           for (int j = 0; j < NT; ++j)
                              g[e][j] -= q_best[e] * h[j + k];
                        }
                     });

                     if (subidx == RakeWidth - 1)
                        h[NT - Padding + k] = 0;//h_buf[rot + k];
                  });

                  sequence<InstLP>::run([&](int e)
                  {
                     reduced[e] += q_best[e] * q_best[e];
                  });
               }

               __syncthreads();

               if (threadIdx.x == 0)
                  sequence<ILP>::run([&](int k)
                  {
                     h_buf[P + rot + k] = h[k];
                  });
            }
         }
         __syncthreads();
      }


      sequence<InstLP>::run([&](int e)
      {
         BlockStoreT(shared.store).Store(g_ptr + (g_base + e * InstPerBlock) * Pitch, g[e]);
         if (reduced[e] > 0.5) gns[g_idx(e)] = -1;
      });

      __syncthreads();

      if (step == 0 && check) break;
   }
}

#if 0
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
