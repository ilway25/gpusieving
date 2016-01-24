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

   p.minimize();
   return {p, p.norm()};
}

frowvec ToVector(const Point& p)
{
   frowvec v(N);
   for (int i = 0; i < N; ++i)
      v[i] = p[i + 1] - p[0];
   return v;
}

pair<Point, Norm> Rectify(const Point& p)
{
   auto q = ToVector(p);
   auto r = round(q);
   auto s = q - r;         // Don't know why this is wrong: q -= r
   auto d = dot(s, s);     // Sanity check

   if (d > 10)
   {
      cout << "Something wrong with" << endl << q << endl << r << endl;
      for (int i = 0; i < P; ++i)
         cout << p[i] << ' ';
      cout << endl;
   }

   return FromVector(r);
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

void List::CopyFromAsync(const List& that, int size, cudaStream_t stream, int offset1, int offset2)
{
   CubDebugExit(cudaMemcpyAsync(points + offset1, that.points + offset2, sizeof(Point) * size, cudaMemcpyDefault, stream));
   CubDebugExit(cudaMemcpyAsync(norms + offset1, that.norms + offset2, sizeof(Norm) * size, cudaMemcpyDefault, stream));
}

void List::CopyFrom(const List& that, int size, int offset1, int offset2)
{
   CubDebugExit(cudaMemcpy(points + offset1, that.points + offset2, sizeof(Point) * size, cudaMemcpyDefault));
   CubDebugExit(cudaMemcpy(norms + offset1, that.norms + offset2, sizeof(Norm) * size, cudaMemcpyDefault));
}

void List::Print(int size, string header)
{
   auto ps = points;
   auto ns = norms;

   // TODO: Clean up
   if (_gpu != -1)
   {
      ps = new Point[size];
      ns = new Norm[size];

      CubDebugExit(cudaSetDevice(_gpu));
      CubDebugExit(cudaMemcpy(ps, points, size * sizeof(Point), cudaMemcpyDefault));
      CubDebugExit(cudaMemcpy(ns, norms, size * sizeof(Norm), cudaMemcpyDefault));
   }

   cout << header << ":\n";
   for (int i = 0; i < size; ++i)
   {
      auto v = ToVector(ps[i]);
      cout << ' ' << header << ": ";
      printf("%d (%.3f) = ", i, ns[i]);
      v.head(16).raw_print();
   }

   if (_gpu != -1)
   {
      delete[] ps;
      delete[] ns;
   }
}

void List::Check(const fmat& B, int size, string header)
{
   auto ps = points;
   auto ns = norms;

   // TODO: Clean up
   if (_gpu != -1)
   {
      ps = new Point[size];
      ns = new Norm[size];

      CubDebugExit(cudaSetDevice(_gpu));
      CubDebugExit(cudaMemcpy(ps, points, size * sizeof(Point), cudaMemcpyDefault));
      CubDebugExit(cudaMemcpy(ns, norms, size * sizeof(Norm), cudaMemcpyDefault));
   }

   fmat M(size, N);

   // cout << "1" << endl;
   for (int i = 0; i < size; ++i)
      M.row(i) = ToVector(ps[i]);

   // cout << "2" << endl;

   // fmat sol = solve(B.t(), M.t());
   fmat sol = M * B;
   // cout << "3" << endl;

   int cnt = 0;
   for (int i = 0; i < size && cnt < 5; ++i)
   {
      bool wrong = false;
      for (int j = 0; j < N; ++j)
         wrong = wrong || (abs(sol(i, j) - std::round(sol(i, j))) > 0.1);

      if (wrong)
      {
         ++cnt;
         cout << header << " (" << i << ")\t";
         M.row(i).head(16).raw_print();
         cout << "  =>\t";
         sol.row(i).head(16).raw_print();
      }
   }
   // cout << "4" << endl;

   if (_gpu != -1)
   {
      delete[] ps;
      delete[] ns;
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
      L2[i].InitGPU(3500000, i);

      CubDebugExit(cudaSetDevice(i));
      CubDebugExit(cudaStreamCreate(&streams[i]));
   }

   cubInit(streams);

   // Prepare S
   best_norm = 1e100;
   for (int i = 0; i < N; ++i)
   {
      tie(S.points[i], S.norms[i]) = FromVector(_B.row(i));
      auto norm = dot(_B.row(i), _B.row(i));
      if (norm < best_norm)
      {
         best_norm = norm;
         shortest_vec = _B.row(i);
      }
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
   _Binv = _B.i();
}

void GSieve::Start()
{
   CubDebugExit(cudaSetDevice(0));

   Point* points;
   Norm*  norms;
   CubDebugExit(cudaMallocHost(&points, sizeof(Point) * NumSamples));
   CubDebugExit(cudaMallocHost(&norms, sizeof(Norm) * NumSamples));

   int *new_Lsize;
   CubDebugExit(cudaMallocHost(&new_Lsize, sizeof(int) * NGPUS));
   new_Lsize[0] = 0;

   int min_L = 0; // Min list from last iteration

   for (int iterations = 0; iterations < 3000; ++iterations)
   {
      cout << "====== Iteration " << iterations << " ======" << endl;

      // cout << "ListS: ";
      // for (int i = 0; i < NGPUS; ++i)
      //    cout << Lsize[i] << ' ';
      // cout << endl;

      CubDebugExit(cudaSetDevice(0));
      GenerateSamples();

      // Sort samples descending
      Q2[0].CopyFrom(Q[0], NumSamples);
      SortPairsDescending(Q2[0].norms, Q[0].norms, Q2[0].points, Q[0].points, NumSamples);

      // Sort list ascending
      CubDebugExit(cudaSetDevice(min_L));
      L2[min_L].CopyFrom(L[min_L], new_Lsize[min_L]);
      SortPairs(L2[min_L].norms, L[min_L].norms, L2[min_L].points, L[min_L].points, new_Lsize[min_L], min_L);

      // Copy GPU 0 samples to CPU to distribute later in Step 0
      CubDebugExit(cudaSetDevice(0));
      CubDebugExit(cudaMemcpy(points, Q[0].points, sizeof(Point) * NumSamples, cudaMemcpyDefault));
      CubDebugExit(cudaMemcpy(norms, Q[0].norms, sizeof(Norm) * NumSamples, cudaMemcpyDefault));

      for (int i = 0; i < NGPUS; ++i)
      {
         CubDebugExit(cudaSetDevice(i));

         // Distribute
         if (i != 0)
         {
            CubDebugExit(cudaMemcpyAsync(Q[i].points, points, sizeof(Point) * NumSamples, cudaMemcpyDefault, streams[i]));
            CubDebugExit(cudaMemcpyAsync(Q[i].norms, norms, sizeof(Norm) * NumSamples, cudaMemcpyDefault, streams[i]));
         }
         CubDebugExit(cudaMemsetAsync(L[i].points + Lsize[i], 0, 1024 * N, streams[i]));
         CubDebugExit(cudaMemsetAsync(L[i].norms + Lsize[i], 0, 1024, streams[i]));
         CubDebugExit(cudaMemsetAsync(Q[i].norms + NumSamples, 0, 1024, streams[i]));
         CubDebugExit(cudaMemsetAsync(Q2[i].norms + NumSamples, 0, 1024, streams[i]));

         // Q[i].Print(NumSamples, "Q-before-" + to_string(i));

         reduce<0><<<GridDim, BlockDim, 0, streams[i]>>>(Q[i].points, Q[i].norms, NumSamples, L[i].points, L[i].norms, Lsize[i]);

         // Q[i].Print(NumSamples, "Q-" + to_string(i));

         Q2[i].CopyFromAsync(Q[i], NumSamples, streams[i]);
         reduce<1><<<GridDim, BlockDim, 0, streams[i]>>>(Q2[i].points, Q2[i].norms, NumSamples, Q[i].points, Q[i].norms, NumSamples);

         // Q2[i].Print(NumSamples, "Q2-" + to_string(i));

         reduce<2><<<GridDim, BlockDim, 0, streams[i]>>>(L[i].points, L[i].norms, Lsize[i], Q2[i].points, Q2[i].norms, NumSamples);

         // L[i].Print(Lsize[i], "L-" + to_string(i));

         TransformInputIterator<bool, NotReduced, Norm*> itr1(L[i].norms, NotReduced());
         PartitionAsync(L[i].points, itr1, L2[i].points, Lsize[i], i);
         SelectIfAsync(L[i].norms, L[i].norms, Lsize[i], NotReduced(), i);
         GetSelectedSizeAsync(&new_Lsize[i], i);

         hostQ[i].CopyFromAsync(Q2[i], NumSamples, streams[i]);
      }

      for (int i = 0; i < NGPUS; ++i)
      {
         CubDebugExit(cudaSetDevice(i));
         CubDebugExit(cudaStreamSynchronize(streams[i]));
      }

      for (int i = 0; i < NGPUS; ++i)
      {
         Q[i].Check(_Binv, NumSamples, "Q-" + to_string(i));
         Q2[i].Check(_Binv, NumSamples, "Q2-" + to_string(i));
         L[i].Check(_Binv, Lsize[i], "L-" + to_string(i));
      }

      // Put reduced vectors (INCLUDING collisions) onto stack
      for (int i = 0; i < NGPUS; ++i)
      {
         CubDebugExit(cudaSetDevice(i));

         int amount = Lsize[i] - new_Lsize[i];
         // S.CopyFrom(L[i], amount, Ssize, new_Lsize[i]);
         CubDebugExit(cudaMemcpy(L[i].points, L2[i].points, sizeof(Point) * new_Lsize[i], cudaMemcpyDefault));
         CubDebugExit(cudaMemcpy(S.points + Ssize, L2[i].points + new_Lsize[i], sizeof(Point) * amount, cudaMemcpyDefault));

         // Recalculate norm
         for (int k = 0; k < amount; ++k)
           S.norms[Ssize + k] = S.points[Ssize + k].norm();

         Ssize += amount;
         Lsize[i] = new_Lsize[i];
      }

      // Remove collisions from stack
      auto mid = partition(S.points, S.points + Ssize, [] (Point n) { return NotReduced()(n.norm()); } );
      partition(S.norms, S.norms + Ssize, [] (Norm n) { return NotReduced()(n); } );
      Ssize = mid - S.points;

      cout << "NLS: ";
      for (int i = 0; i < NGPUS; ++i)
         cout << Lsize[i] << ' ';
      cout << endl;

      int cnt_r = 0, cnt_nr = 0;
      for (int i = 0; i < NumSamples; ++i)
      {
         // Not reduced -> collect and add to one list
         if (all_of(hostQ, hostQ + NGPUS, [=](const List& l) { return NotReduced()(l.norms[i]); }))
         {
            points[cnt_nr] = hostQ[0].points[i];
            norms[cnt_nr] = hostQ[0].norms[i];
            ++cnt_nr;
         }
         else // Reduced -> throw away collisions, add min rep to stack
         {
            Norm real_norms[NGPUS] {};
            for (int j = 0; j < NGPUS; ++j)
            {
               hostQ[j].points[i].minimize();
               hostQ[j].norms[i] = hostQ[j].points[i].norm(); // Some are -1

               real_norms[j] = hostQ[j].norms[i] + P * hostQ[j].points[i][0] * hostQ[j].points[i][0];
            }

            auto itmin = min_element(real_norms, real_norms + NGPUS);
            int argmin = itmin - real_norms;
            if (NotReduced()(*itmin)) // Actually check for collision
            {
               tie(S.points[Ssize], S.norms[Ssize]) = Rectify(hostQ[argmin].points[i]);

               // cout << "put to" << Ssize << ' ' << i << ' ' << S.norms[Ssize] << endl;

               if (*itmin < best_norm)
               {
                  best_norm = *itmin;
                  shortest_vec = ToVector(S.points[Ssize]);

                  // found_time = system_clock::now();
               }
               ++Ssize;
               ++cnt_r;
            }
         }
      }
      cout << "NR: " << cnt_nr << "  R: " << cnt_r << "  C: " << NumSamples - cnt_nr - cnt_r << endl;
      cout << "S:" << Ssize << endl;

      int min_L = min_element(Lsize, Lsize + NGPUS) - Lsize;
      cout << "Append to List " << min_L << endl;
      CubDebugExit(cudaSetDevice(min_L));
      CubDebugExit(cudaMemcpy(L[min_L].points + Lsize[min_L], points, sizeof(Point) * cnt_nr, cudaMemcpyDefault));
      CubDebugExit(cudaMemcpy(L[min_L].norms + Lsize[min_L], norms, sizeof(Norm) * cnt_nr, cudaMemcpyDefault));
      Lsize[min_L] += cnt_nr;

      printf("Min Norm = %.3f\n", best_norm);
      cout << '[';
      for (int i = 0; i < N; ++i)
         cout << shortest_vec[i] << ' ';
      cout << ']' << endl;

      // for (int i = 0; i < NGPUS; ++i)
      //    L[i].Print(Lsize[i], "L" + to_string(i));
      // S.Print(Ssize, "S");

      if (N == 96 && best_norm < 6327000) break;
   }
}

void GSieve::GenerateSamples()
{
   Point points[NumSamples];
   Norm  norms[NumSamples];

   int amount = ::min(NumSamples, Ssize);
   Q[0].CopyFrom(S, amount, 0, Ssize - amount);
   Ssize -= amount;

   for (int i = amount; i < NumSamples; ++i)
   {
      frowvec v(N);
      for (int j = 0; j < N; ++j)
         _sample_stream >> v[j];

      tie(points[i], norms[i]) = FromVector(v);

      float skip;
      _sample_stream >> skip;
   }

   CubDebugExit(cudaMemcpy(Q[0].points + amount, points + amount, sizeof(Point) * (NumSamples - amount), cudaMemcpyDefault));
   CubDebugExit(cudaMemcpy(Q[0].norms + amount, norms + amount, sizeof(Norm) * (NumSamples - amount), cudaMemcpyDefault));
}

/*
{ // REF
   List L1, L2;
   L1.InitHost(NumSamples);
   L2.InitHost(NumSamples);

   CubDebugExit(cudaMemcpy(L1.points, Q[0].points, sizeof(Point) * NumSamples, cudaMemcpyDefault));
   CubDebugExit(cudaMemcpy(L1.norms, Q[0].norms, sizeof(Norm) * NumSamples, cudaMemcpyDefault));

   CubDebugExit(cudaMemcpy(L2.points, Q[0].points, sizeof(Point) * NumSamples, cudaMemcpyDefault));
   CubDebugExit(cudaMemcpy(L2.norms, Q[0].norms, sizeof(Norm) * NumSamples, cudaMemcpyDefault));

   GoldenReduce(L1.points, L1.norms, NumSamples, L2.points, L2.norms, NumSamples);

   L1.Print(100, "L1");
}
*/

void GSieve::GoldenReduce(Point* gs, Norm* gns, size_t gsize, const Point* hs, const Norm* hns, size_t hsize)
{
   for (int i = 0; i < gsize; ++i)
   {
      Point& g = gs[i];
      Norm&  gg = gns[i];

      float min_norm = gg + P * g[0] * g[0];
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
