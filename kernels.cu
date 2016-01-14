#include <cub/cub.cuh>

#include "kernels.cuh"

__device__ float tmp;

const int NumPrefetch = CUB_QUOTIENT_FLOOR(4 * BlockDim, Pitch);

using shared_t = union
{
   float block[BlockDim][4];
   float sep[NumPrefetch][RakeWidth][NT];
   float linear[NumPrefetch][Pitch];
};

__global__
void reduce(Point* gs, Norm* gns, size_t g_size, const Point* hs, const Norm* hns, size_t h_size)
{
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

      BlockLoadT(shared.load).Load(g_in + g_base * Pitch, g);
      gg = gns[g_idx];

      float min_norm = gg + P * (g_in + g_base * Pitch)[0] * (g_in + g_base * Pitch)[0];

      __shared__ alignas(128) shared_t prefetch;
      __shared__ float prefetch_n[BlockDim];

      for (int h_base = 0; h_base < h_size; h_base += NumPrefetch)
      {
         BlockLoadVT(shared.loadv).Load(h_in + h_base * Pitch, prefetch.block[threadIdx.x]);
         prefetch_n[threadIdx.x] = hns[h_base + threadIdx.x];
         __syncthreads();

         for (int i = 0; i < NumPrefetch && h_base + i < h_size; ++i)
         {
            const int h_idx = h_base + i;

            // g_buf has no zero padding
            using sep = float[RakeWidth][NT];
            __shared__ float h_buf[Pitch * 2];

            if (threadIdx.x < P)
            {
               h_buf[threadIdx.x] = prefetch.linear[i][threadIdx.x];
               h_buf[threadIdx.x + P] = 0;
            }

            if (threadIdx.x < Pitch * 2 - P * 2)
               h_buf[threadIdx.x + 2 * P] = 0;

            float hh = prefetch_n[i];

            __syncthreads();

            for (int rot = 0; rot < P; ++rot)
            {
               float q_best {};
               float gh {};

               float h[NT];
               for (int j = 0; j < NT; ++j)
                  h[j] = (*(sep*)(&h_buf[rot]))[subidx][j];

               for (int j = 0; j < NT; ++j)
                  gh += g[j] * h[j];

               for (int j = 1; j < RakeWidth; j *= 2)
                  gh += __shfl_xor(gh, j);

               for (int j = 0; j < NT; ++j)
               {
                  float uu = gg + P * g[j] * g[j],
                        uv = gh + P * g[j] * h[j],
                        vv = hh + P * h[j] * h[j];

                  float q = rintf(uv / uu);
                  float new_norm = uu - 2 * q * uv + q * q * vv;

                  if (new_norm < min_norm && q != 0 && subidx * NT + j < P)
                  {
                     // printf("%d, %d, %d (%d) -> %.0f, %.0f, %.0f -> %.0f (%.0f, %.0f)\n",
                     //    g_idx, h_idx, subidx * NT + j, rot,
                     //    uu, uv, vv, q, new_norm, min_norm);

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

               for (int j = 0; j < NT; ++j)
                  g[j] -= q_best * h[j];
               gg += q_best * q_best * hh - q_best * 2 * gh;

               // if (q_best != 0 && subidx == 0)
               //    printf("%d %d %d %f\n", g_idx, h_idx, rot, q_best);

               if (threadIdx.x == 0)
                  h_buf[P + rot] = h[0];

               __syncthreads();
            }
         }
      }

      BlockStoreT(shared.store).Store(g_ptr + g_base * Pitch, g);
      gns[g_idx] = gg;
   }
}
