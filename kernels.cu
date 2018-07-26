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

    template <typename Lambda>
    static __forceinline__ __device__ void reverse(const Lambda& f) {
        f(N-1);
        sequence<N-1>::reverse(f);
    }
};


template <>
struct sequence<0> {
    template <typename Lambda>
    static __forceinline__ __device__ void run(const Lambda& f) {}
    template <typename Lambda>
    static __forceinline__ __device__ void reverse(const Lambda& f) {}
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


   for (int g_base = blockIdx.x * InstPerBlock; g_base < g_size; g_base += GridDim * InstPerBlock)
   {
      const auto g_idx = g_base + subinst;

      float g[NT], gg;
      float reduced {}; // Flag: 0 <-> Not reduced.

      float min_norm;

      BlockLoadT(shared.load).Load(g_in + g_base * Pitch, g);
      gg = gns[g_idx];
      auto t = (g_in + g_base * Pitch)[0];
      min_norm = gg + P * t * t;

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

               float q_best {};
               float gh[ILP] {};

               float h[NT + (ILP - 1)];
               for (int j = 0; j < NT; ++j)
                  h[j] = (*(volatile sep*)(&h_buf[rot]))[subidx][j];

               sequence<ILP - 1>::run([&](int k)
               {
                  h[NT + k] = h_buf[rot + (subidx + 1) * NT + k]; // 小心出界
               });

               sequence<ILP>::run([&](int k)
               {
                  for (int j = 0; j < NT; ++j)
                     gh[k] += g[j] * h[j + k];

                  if (subidx == RakeWidth - 1)
                     h[NT - Padding + k] = h_buf[rot + k];
               });

               for (int j = 1; j < RakeWidth; j *= 2)
                  sequence<ILP>::run([&](int k)
                  {
                     gh[k] += __shfl_xor(gh[k], j);
                  });

               int from {};
               for (int j = 0; j < Times; ++j) // j 可以 1 至 NT
               {
                  float uu = gg + (P * g[j]) * g[j];

                  sequence<ILP>::run([&](int k)
                  {
                     float uv = gh[k] + (P * g[j]) * h[j + k],
                           vv = hh + P * h[j + k] * h[j + k];

                     float q = rintf(uv / uu);

                     if (step == 1 && gg < 0) q = 0;
                     if (step == 1 && g_idx == h_idx && rot == 0 && k == 0) q = 0;

                     float new_norm = uu + q * (q * vv - 2 * uv);
                     if (new_norm < min_norm) // 若 j 快到 NT，要加 && subidx * NT + j < P)
                     {
                        q_best = q;
                        from = k + 1; // 因為 k 的預設值是 0，後會面多做事
                     }
                     min_norm = min(new_norm, min_norm);
                  });
               }

               for (int j = 1; j < RakeWidth; j *= 2)
               {
                  float min_norm_t = __shfl_xor(min_norm, j);
                  float q_best_t = __shfl_xor(q_best, j);
                  int   from_t = __shfl_xor(from, j);

                  // 最好加後面那句避免非常非常小的機率錯誤
                  if (min_norm_t < min_norm) // || min_norm_t == min_norm && (subidx ^ j) >= subidx)
                  {
                     q_best = q_best_t;
                     from = from_t;
                  }
                  min_norm = min(min_norm_t, min_norm);
               }

               if (step == 0 || __any(q_best != 0)) // 這行舊版沒有
               {
                  sequence<ILP>::reverse([&](int k) // 倒著跑
                  {
                     if (from == k + 1)
                     {
                        gg += q_best * (q_best * hh - 2 * gh[k]);
                        for (int j = 0; j < NT; ++j)
                           g[j] -= q_best * h[j + k];
                     }

                     if (subidx == RakeWidth - 1)
                        h[NT - Padding + k] = 0;
                  });

                  reduced += q_best * q_best;
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

      BlockStoreT(shared.store).Store(g_ptr + g_base * Pitch, g);
      if (reduced > 0.5) gns[g_idx] = -1;

      __syncthreads();

      if (step == 0 && check) break;
   }
}

