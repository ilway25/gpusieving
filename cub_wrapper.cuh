#pragma once

#define CUB_STDERR

#include <cub/cub.cuh>

using namespace cub;

void cubInit(cudaStream_t* streams);

// Single Device
template <typename Key, typename Value>
void SortPairsDescending(Key* d_keys_in, Key* d_keys_out, Value* d_values_in, Value* d_values_out, int num_items);

template <typename Key, typename Value>
void SortPairs(Key* d_keys_in, Key* d_keys_out, Value* d_values_in, Value* d_values_out, int num_items, int dev);

// Streams
template <typename InputIterator, typename FlagIterator, typename OutputIterator>
void PartitionAsync(InputIterator d_in, FlagIterator d_flags, OutputIterator d_out, int num_items, int dev);

template <typename InputIterator, typename OutputIterator, typename SelectOp>
void SelectIfAsync(InputIterator d_in, OutputIterator d_out, int num_items, SelectOp select_op, int dev);

void GetSelectedSizeAsync(int* n, int dev);
