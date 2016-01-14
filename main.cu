#include <iostream>

#include "gsieve.cuh"

using namespace std;

int main()
{
   GSieve gs("basis/basis102p", cin);

   gs.Start();
}