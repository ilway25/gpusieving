#include "cub_wrapper.cuh"
#include "kernels.cuh"

int *d_num_selected[NGPUS];

typedef TransformInputIterator<bool, NotReduced, Norm*> TII;

template
void PartitionAsync<Point*, TII, Point*>(Point*, TII, Point*, int, int);

template
void SelectIfAsync<Norm*, Norm*, NotReduced>(Norm*, Norm*, int, NotReduced, int);

template
void SortPairs<unsigned, Point>(unsigned*, unsigned*, Point*, Point*, int, int);

template
void SortPairsDescending<unsigned, Point>(unsigned*, unsigned*, Point*, Point*, int);

void*  d_temp = 0;
size_t temp_size = 0;

void*  temps[NGPUS];
size_t sizes[NGPUS];

cudaStream_t* streams;

void cubInit(cudaStream_t* ss)
{
   streams = ss;
   for (int i = 0; i < NGPUS; ++i)
   {
      CubDebugExit(cudaSetDevice(i));
      sizes[i] = 1024 * 1024;
      CubDebugExit(cudaMalloc(&temps[i], sizes[i]));
      CubDebugExit(cudaMalloc(&d_num_selected[i], sizeof(int)));
   }
}

void GetSelectedSizeAsync(int* n, int dev)
{
   CubDebugExit(cudaMemcpyAsync(n, d_num_selected[dev], sizeof(int), cudaMemcpyDefault, streams[dev]));
}

// Warppers for CUB functions
template <typename InputIterator, typename FlagIterator, typename OutputIterator>
void PartitionAsync(InputIterator d_in, FlagIterator d_flags, OutputIterator d_out, int num_items, int dev)
{
   size_t bytes;
   CubDebugExit(DevicePartition::Flagged(0, bytes, d_in, d_flags, d_out, d_num_selected[dev], num_items, streams[dev]));
   if (bytes > sizes[dev])
   {
      CubDebugExit(cudaFree(temps[dev]));
      CubDebugExit(cudaMalloc(&temps[dev], bytes));
      sizes[dev] = bytes;
   }
   CubDebugExit(DevicePartition::Flagged(temps[dev], bytes, d_in, d_flags, d_out, d_num_selected[dev], num_items, streams[dev]));
}

template <typename InputIterator, typename OutputIterator, typename SelectOp>
void SelectIfAsync(InputIterator d_in, OutputIterator d_out, int num_items, SelectOp select_op, int dev)
{
   size_t bytes;
   CubDebugExit(DeviceSelect::If(0, bytes, d_in, d_out, d_num_selected[dev], num_items, select_op, streams[dev]));
   if (bytes > sizes[dev])
   {
      CubDebugExit(cudaFree(temps[dev]));
      CubDebugExit(cudaMalloc(&temps[dev], bytes));
      sizes[dev] = bytes;
   }
   CubDebugExit(DeviceSelect::If(temps[dev], bytes, d_in, d_out, d_num_selected[dev], num_items, select_op, streams[dev]));
}

template <typename Key, typename Value>
void SortPairs(Key* d_keys_in, Key* d_keys_out, Value* d_values_in, Value* d_values_out, int num_items, int dev)
{
   size_t bytes;
   CubDebugExit(cub::DeviceRadixSort::SortPairs(0, bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, num_items));

   if (bytes > sizes[dev])
   {
      CubDebugExit(cudaFree(temps[dev]));
      CubDebugExit(cudaMalloc(&temps[dev], bytes));
      sizes[dev] = bytes;
   }

   CubDebugExit(cub::DeviceRadixSort::SortPairs(temps[dev], bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, num_items));
}

template <typename Key, typename Value>
void SortPairsDescending(Key* d_keys_in, Key* d_keys_out, Value* d_values_in, Value* d_values_out, int num_items)
{
   size_t bytes;
   CubDebugExit(cub::DeviceRadixSort::SortPairs(0, bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, num_items));

   if (bytes > temp_size)
   {
      CubDebugExit(cudaFree(d_temp));
      CubDebugExit(cudaMalloc(&d_temp, bytes));
      temp_size = bytes;
   }

   CubDebugExit(cub::DeviceRadixSort::SortPairsDescending(d_temp, bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, num_items));
}