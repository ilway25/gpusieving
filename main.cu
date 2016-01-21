#include <iostream>
#include <fstream>

#include "gsieve.cuh"

using namespace std;

int main()
{
   ifstream fin("sample/_samples102p");
   GSieve gs("basis/basis102p", fin);

   gs.Start();
}