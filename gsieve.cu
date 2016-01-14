#include <fstream>
#include <sstream>
#include <regex>
#include <vector>
#include <cassert>

#include "gsieve.cuh"

#define DEBUG

#include <cub/util_debug.cuh>

pair<Point, Norm> FromVector(const frowvec& v)
{
   Point p;
   auto t = sum(v) / P;
   p[0] = -t;
   for (int i = 0; i < N; ++i)
      p[i + 1] = v[i] - t;

   Norm n = dot(v, v) - P * t * t;
   return {p, n};
}

frowvec ToVector(const Point& p, Norm n)
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
         auto v = ToVector(ps[i], ns[i]);
         cout << i << " (" << int(dot(v, v)) << ")\t";
         v.raw_print();
      }
   }
   else
   {
      cout << header << ":\n";
      for (int i = 0; i < size; ++i)
      {
         auto v = ToVector(points[i], norms[i]);
         cout << i << " (" << int(dot(v, v)) << ")\t";
         v.raw_print();
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
      Q[i].InitGPU(NumSamples * 5, i);
      Q2[i].InitGPU(NumSamples * 5, i);
      L[i].InitGPU(3500000, i);

      CubDebugExit(cudaSetDevice(i));
      CubDebugExit(cudaStreamCreate(&streams[i]));
   }

   // Prepare S
   for (int i = 0; i < N; ++i)
   {
      auto p = FromVector(_B.row(i));

      S.points[i] = p.first;
      S.norms[i] = p.second;

      // conv_to<irowvec>::from(_B.row(i)).head(16).raw_print();
   }

   // S.Print(10, "S");
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

   CubDebugExit(cudaMemcpy(Q[0].points, S.points, 50 * sizeof(Point), cudaMemcpyDefault));
   CubDebugExit(cudaMemcpy(Q[0].norms, S.norms, 50 * sizeof(Norm), cudaMemcpyDefault));
   CubDebugExit(cudaMemcpy(Q2[0].points, S.points + 50, 50 * sizeof(Point), cudaMemcpyDefault));
   CubDebugExit(cudaMemcpy(Q2[0].norms, S.norms + 50, 50 * sizeof(Norm), cudaMemcpyDefault));

   // CubDebugExit(cudaMemset(Q[0].points + 50, 0, 102400));
   // CubDebugExit(cudaMemset(Q[0].norms + 50, 0, 1024));


   reduce<<<BlockDim, GridDim>>>(Q[0].points, Q[0].norms, 50, Q2[0].points, Q2[0].norms, 50);

   // Q[0].Print(50, "Q");

   cudaDeviceSynchronize();

   cout << "----------------------------------------------------------" << endl;

   // S.Print(50, "Q");
   GoldenReduce(S.points, S.norms, 50, S.points + 50, S.norms + 50, 50);
   // S.Print(50, "Q");

}

void GSieve::GoldenReduce(Point* gs, Norm* gns, size_t g_size, const Point* hs, const Norm* hns, size_t h_size)
{
   for (int i = 0; i < g_size; ++i)
   {
      Point& g = gs[i];
      Norm&  gg = gns[i];

      float min_norm = gg + P * g[0] * g[0];
      // cout << min_norm << endl;
      for (int j = 0; j < h_size; ++j)
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
                  printf("%d, %d, %d (%d) -> %.0f, %.0f, %.0f -> %.0f (%.0f, %.0f)\n",
                     i, j, k, rot, uu, uv, vv, m, new_norm, min_norm);
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
