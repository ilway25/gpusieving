#include <iostream>
#include <armadillo>

using namespace std;
using namespace arma;

namespace sc = std::chrono;

#include "kernels.cuh"

struct List
{
   void InitHost(int size);
   void InitGPU(int size, int gpu);
   ~List();

   void Print(int size, string header = {});
   void Check(const fmat& B, int size, string header = {});

   void CopyFromAsync(const List& that, int size, cudaStream_t, int offset1 = 0, int offset2 = 0);
   void CopyFrom(const List& that, int size, int offset1 = 0, int offset2 = 0);

   Point* points;
   Norm* norms;

   int _gpu = -1;
};


class GSieve
{
public:
   GSieve(string basis, istream& sample_stream);
   ~GSieve();

   void Start();
   void GenerateSamples();

   void GoldenReduce(Point*, Norm*, size_t, const Point*, const Norm*, size_t);

   void Save(string filename);
   void Load(string filename);

private:
   void ReadBasis(string filename);

   istream& _sample_stream;

   fmat     _B;
   fmat     _Binv;

   List S, hostQ[NGPUS];                // Host
   List L[NGPUS], Q[NGPUS], Q2[NGPUS], L2[NGPUS];  // Gpu

   int Ssize;
   int Lsize[NGPUS] {};

   cudaStream_t streams[NGPUS];

   Norm    best_norm;
   frowvec shortest_vec;

   int iterations {0};
   sc::system_clock::time_point found_time, start_time;
   sc::milliseconds             duration {0};
};

template<typename T>
void print_duration(std::ostream& out, T dur)
{
    auto diff = sc::duration_cast<sc::seconds>(dur).count();
    auto const secs = diff % 60;
    diff /= 60;
    auto const mins = diff % 60;
    diff /= 60;
    auto const hours = diff % 24;
    diff /= 24;
    auto const days = diff;

    bool printed_earlier = false;
    if (days >= 1) {
        printed_earlier = true;
        out << days << (1 != days ? " days" : " day") << ' ';
    }
    if (printed_earlier || hours >= 1) {
        printed_earlier = true;
        out << hours << (1 != hours ? " hours" : " hour") << ' ';
    }
    if (printed_earlier || mins >= 1) {
        printed_earlier = true;
        out << mins << (1 != mins ? " minutes" : " minute") << ' ';
    }
    if (printed_earlier || secs >= 1) {
        printed_earlier = true;
        out << secs << (1 != secs ? " seconds" : " second") << ' ';
    }
}
