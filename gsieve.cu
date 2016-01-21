#include <fstream>
#include <sstream>
#include <regex>
#include <vector>
#include <cassert>

#include "gsieve.cuh"
#include "cub_wrapper.cuh"

#define DEBUG

#include <cub/util_debug.cuh>

pair<Point, Norm> FromVector(const frowvec& v)
{
   Point p;
   auto t = sum(v) / P;
   p[0] = -t;
   for (int i = 0; i < N; ++i)
      p[i + 1] = v[i] - t;
   for (int i = P; i < Pitch; ++i)
      p[i] = 0;

   Norm n = dot(v, v) - P * t * t;
   return {p, n};
}

frowvec ToVector(const Point& p)
{
   frowvec v(N);
   for (int i = 0; i < N; ++i)
      v[i] = p[i + 1] - p[0];
   return v;
}

void List::InitHost(int size)
{
   CubDebugExit(cudaMallocHost(&points, sizeof(Point) * size));
   CubDebugExit(cudaMallocHost(&norms, sizeof(Norm) * size));
}

void List::InitGPU(int size, int gpu)
{
   _gpu = gpu;
   cudaSetDevice(gpu);
   CubDebugExit(cudaMalloc(&points, sizeof(Point) * size));
   CubDebugExit(cudaMalloc(&norms, sizeof(Norm) * size));
}

void List::Print(int size, string header)
{
   // TODO: Clean up
   if (_gpu != -1)
   {
      Point ps[size];
      Norm ns[size];

      CubDebugExit(cudaSetDevice(_gpu));
      CubDebugExit(cudaMemcpy(ps, points, size * sizeof(Point), cudaMemcpyDefault));
      CubDebugExit(cudaMemcpy(ns, norms, size * sizeof(Norm), cudaMemcpyDefault));

      cout << header << ":\n";
      for (int i = 0; i < size; ++i)
      {
         auto v = ToVector(ps[i]);
         cout << i << " (" << int(ns[i]) << ")\t";
         v.head(16).raw_print();
      }
   }
   else
   {
      cout << header << ":\n";
      for (int i = 0; i < size; ++i)
      {
         auto v = ToVector(points[i]);
         cout << i << " (" << int(norms[i]) << ")\t";
         v.head(16).raw_print();
      }
   }
}

List::~List()
{
   if (_gpu != -1)
   {
      CubDebugExit(cudaSetDevice(_gpu));
      CubDebugExit(cudaFree(points));
      CubDebugExit(cudaFree(norms));
   }
   else
   {
      CubDebugExit(cudaFreeHost(points));
      CubDebugExit(cudaFreeHost(norms));
   }
}

GSieve::GSieve(string basis, istream& sample_stream)
   : _sample_stream(sample_stream)
{
   ReadBasis(basis);

   S.InitHost(100000);
   for (int i = 0; i < NGPUS; ++i)
   {
      hostQ[i].InitHost(NumSamples);

      Q[i].InitGPU(NumSamples + 65536, i);
      Q2[i].InitGPU(NumSamples + 65536, i);
      L[i].InitGPU(3500000, i);

      CubDebugExit(cudaSetDevice(i));
      CubDebugExit(cudaStreamCreate(&streams[i]));
   }

   cubInit(streams);

   // Prepare S
   for (int i = 0; i < N; ++i)
   {
      auto p = FromVector(_B.row(i));
      S.points[i] = p.first;
      S.norms[i] = p.second;
   }
   Ssize = N;
}

GSieve::~GSieve()
{
   for (int i = 0; i < NGPUS; ++i)
   {
      CubDebugExit(cudaSetDevice(i));
      CubDebugExit(cudaStreamDestroy(streams[i]));
   }
}

void GSieve::ReadBasis(string filename)
{
   ifstream fin(filename);
   assert(fin);

   stringstream ss;
   ss << fin.rdbuf();
   auto str = ss.str();

   regex pat{"-?\\d+"}; // Extract all numbers
   sregex_token_iterator p(begin(str), end(str), pat);

   vector<float> nums;
   transform(p, {}, back_inserter(nums), [](string s) { return stof(s); } );
   assert(nums.size() == N * N);

   _B = reshape(fmat(nums), N, N).t();
}

void GSieve::Start()
{
   CubDebugExit(cudaSetDevice(0));

   Point* points;
   Norm*  norms;
   CubDebugExit(cudaMallocHost(&points, sizeof(Point) * NumSamples));
   CubDebugExit(cudaMallocHost(&norms, sizeof(Norm) * NumSamples));

   for (int iterations = 0; iterations < 1; ++iterations)
   {
      int new_Lsize[NGPUS];

      cout << "====== Iteration " << iterations << " ======" << endl;

      CubDebugExit(cudaSetDevice(0));
      GenerateSamples(); // Current WRONG

      // Copy GPU 0 samples to CPU to distribute later in Step 0
      CubDebugExit(cudaSetDevice(0));
      CubDebugExit(cudaMemcpy(points, Q[0].points, sizeof(Point) * NumSamples, cudaMemcpyDefault));
      CubDebugExit(cudaMemcpy(norms, Q[0].norms, sizeof(Norm) * NumSamples, cudaMemcpyDefault));

      // { // REF
      //    List L1, L2;
      //    L1.InitHost(NumSamples);
      //    L2.InitHost(NumSamples);

      //    CubDebugExit(cudaMemcpy(L1.points, Q[0].points, sizeof(Point) * NumSamples, cudaMemcpyDefault));
      //    CubDebugExit(cudaMemcpy(L1.norms, Q[0].norms, sizeof(Norm) * NumSamples, cudaMemcpyDefault));

      //    CubDebugExit(cudaMemcpy(L2.points, Q[0].points, sizeof(Point) * NumSamples, cudaMemcpyDefault));
      //    CubDebugExit(cudaMemcpy(L2.norms, Q[0].norms, sizeof(Norm) * NumSamples, cudaMemcpyDefault));

      //    GoldenReduce(L1.points, L1.norms, NumSamples, L2.points, L2.norms, NumSamples);

      //    L1.Print(100, "L1");
      // }

      {
         CubDebugExit(cudaSetDevice(0));
         CubDebugExit(cudaMemcpy(L[0].points, Q[0].points + 300, sizeof(Point) * 300, cudaMemcpyDefault));
         CubDebugExit(cudaMemcpy(L[0].norms, Q[0].norms + 300, sizeof(Norm) * 300, cudaMemcpyDefault));
         Lsize[0] = 300;

         CubDebugExit(cudaSetDevice(1));
         CubDebugExit(cudaMemcpy(L[1].points, Q[0].points + 600, sizeof(Point) * 300, cudaMemcpyDefault));
         CubDebugExit(cudaMemcpy(L[1].norms, Q[0].norms + 600, sizeof(Norm) * 300, cudaMemcpyDefault));
         Lsize[1] = 300;

         CubDebugExit(cudaSetDevice(2));
         CubDebugExit(cudaMemcpy(L[2].points, Q[0].points + 900, sizeof(Point) * 300, cudaMemcpyDefault));
         CubDebugExit(cudaMemcpy(L[2].norms, Q[0].norms + 900, sizeof(Norm) * 300, cudaMemcpyDefault));
         Lsize[2] = 300;
      }


      for (int i = 0; i < NGPUS; ++i)
      {
         CubDebugExit(cudaSetDevice(i));

         // Distribute
         if (i != 0)
         {
            CubDebugExit(cudaMemcpyAsync(Q[i].points, points, sizeof(Point) * NumSamples, cudaMemcpyDefault, streams[i]));
            CubDebugExit(cudaMemcpyAsync(Q[i].norms, norms, sizeof(Norm) * NumSamples, cudaMemcpyDefault, streams[i]));
         }

         CubDebugExit(cudaMemsetAsync(L[i].norms + Lsize[i], 0, 1024, streams[i]));
         CubDebugExit(cudaMemsetAsync(Q[i].norms + NumSamples, 0, 1024, streams[i]));
         CubDebugExit(cudaMemsetAsync(Q2[i].norms + NumSamples, 0, 1024, streams[i]));

         reduce<0><<<GridDim, BlockDim, 0, streams[i]>>>(Q[i].points, Q[i].norms, NumSamples, L[i].points, L[i].norms, Lsize[i]);

         CubDebugExit(cudaMemcpyAsync(Q2[i].points, Q[i].points, sizeof(Point) * NumSamples, cudaMemcpyDefault, streams[i]));
         CubDebugExit(cudaMemcpyAsync(Q2[i].norms, Q[i].norms, sizeof(Norm) * NumSamples, cudaMemcpyDefault, streams[i]));

         reduce<1><<<GridDim, BlockDim, 0, streams[i]>>>(Q2[i].points, Q2[i].norms, NumSamples, Q[i].points, Q[i].norms, NumSamples);

         // CubDebugExit(cudaMemcpyAsync(Q[i].points, Q2[i].points, sizeof(Point) * NumSamples, cudaMemcpyDefault, streams[i]));
         // CubDebugExit(cudaMemcpyAsync(Q[i].norms, Q2[i].norms, sizeof(Norm) * NumSamples, cudaMemcpyDefault, streams[i]));

         reduce<2><<<GridDim, BlockDim, 0, streams[i]>>>(L[i].points, L[i].norms, Lsize[i], Q2[i].points, Q2[i].norms, NumSamples);

         // Partition 似乎可以輸入輸出相同
         TransformInputIterator<bool, NotReduced, Norm*> itr1(L[i].norms, NotReduced());
         PartitionAsync(L[i].points, itr1, L[i].points, Lsize[i], i);
         SelectIfAsync(L[i].norms, L[i].norms, Lsize[i], NotReduced(), i);
         GetSelectedSizeAsync(&new_Lsize[i], i);

         // Send Q to CPU
         CubDebugExit(cudaMemcpyAsync(hostQ[i].points, Q[i].points, sizeof(Point) * NumSamples, cudaMemcpyDefault, streams[i]));
         CubDebugExit(cudaMemcpyAsync(hostQ[i].norms, Q[i].norms, sizeof(Norm) * NumSamples, cudaMemcpyDefault, streams[i]));
      }

      for (int i = 0; i < NGPUS; ++i)
      {
         CubDebugExit(cudaSetDevice(i));
         CubDebugExit(cudaStreamSynchronize(streams[i]));
      }
   }
   // CubDebugExit(cudaMemcpy(Q[0].points, S.points, 50 * sizeof(Point), cudaMemcpyDefault));
   // CubDebugExit(cudaMemcpy(Q[0].norms, S.norms, 50 * sizeof(Norm), cudaMemcpyDefault));
   // CubDebugExit(cudaMemcpy(Q2[0].points, S.points + 50, 50 * sizeof(Point), cudaMemcpyDefault));
   // CubDebugExit(cudaMemcpy(Q2[0].norms, S.norms + 50, 50 * sizeof(Norm), cudaMemcpyDefault));

   // reduce<<<BlockDim, GridDim>>>(Q[0].points, Q[0].norms, 50, Q2[0].points, Q2[0].norms, 50);

   // // Q[0].Print(50, "Q");

   // cudaDeviceSynchronize();

   // cout << "----------------------------------------------------------" << endl;

   // // S.Print(50, "Q");
   // GoldenReduce(S.points, S.norms, 50, S.points + 50, S.norms + 50, 50);
   // // S.Print(50, "Q");

}

void GSieve::GenerateSamples()
{
   Point points[NumSamples];
   Norm  norms[NumSamples];

   int copysize = ::min(NumSamples, Ssize);

   CubDebugExit(cudaMemcpy(Q[0].points, S.points + Ssize - copysize, sizeof(Point) * copysize, cudaMemcpyDefault));
   CubDebugExit(cudaMemcpy(Q[0].norms, S.norms + Ssize - copysize, sizeof(Norm) * copysize, cudaMemcpyDefault));

   Ssize -= copysize;

   for (int i = copysize; i < NumSamples; ++i)
   {
      frowvec v(N);
      for (int j = 0; j < N; ++j)
         _sample_stream >> v[j];

      auto p = FromVector(v);
      points[i] = p.first;
      norms[i] = p.second;

      float skip;
      _sample_stream >> skip;
   }

   CubDebugExit(cudaMemcpy(Q[0].points + copysize, points + copysize, sizeof(Point) * (NumSamples - copysize), cudaMemcpyDefault));
   CubDebugExit(cudaMemcpy(Q[0].norms + copysize, norms + copysize, sizeof(Norm) * (NumSamples - copysize), cudaMemcpyDefault));
}

void GSieve::GoldenReduce(Point* gs, Norm* gns, size_t gsize, const Point* hs, const Norm* hns, size_t hsize)
{
   for (int i = 0; i < gsize; ++i)
   {
      Point& g = gs[i];
      Norm&  gg = gns[i];

      float min_norm = gg + P * g[0] * g[0];
      // cout << min_norm << endl;
      for (int j = 0; j < hsize; ++j)
      {
         const Point& h = hs[j];
         const Norm  hh = hns[j];

         for (int rot = 0; rot < P; ++rot)
         {
            float gh {};
            for (int k = 0; k < P; ++k)
               gh += g[k] * h[(k + rot) % P];

            float best_m = 0;

            for (int k = 0; k < P; ++k)
            {
               float p = g[k], q = h[(k + rot) % P];

               float uu = gg + P * p * p,
                     uv = gh + P * p * q,
                     vv = hh + P * q * q;

               float m = std::round(uv / uu);
               float new_norm = uu - 2 * m * uv + m * m * vv;

               if (new_norm < min_norm && m != 0)
               {
                  // printf("%d, %d, %d (%d) -> %.0f, %.0f, %.0f -> %.0f (%.0f, %.0f)\n",
                  //    i, j, k, rot, uu, uv, vv, m, new_norm, min_norm);
                  min_norm = new_norm;
                  best_m = m;
               }
            }

            for (int k = 0; k < P; ++k)
               g[k] -= best_m * h[(k + rot) % P];

            gg += best_m * best_m * hh - best_m * 2 * gh;
         }
      }
   }
}
