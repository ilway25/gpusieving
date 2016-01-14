#include <iostream>
#include <armadillo>

using namespace std;
using namespace arma;

#include "kernels.cuh"

struct List
{
   void InitHost(int size);
   void InitGPU(int size, int gpu);
   ~List();

   void Print(int size, string header = {});

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

   void GoldenReduce(Point*, Norm*, size_t, const Point*, const Norm*, size_t);

private:
   void ReadBasis(string filename);

   istream& _sample_stream;

   fmat     _B;

   List S;
   List L[NGPUS], Q[NGPUS], Q2[NGPUS];

   cudaStream_t streams[NGPUS];
};