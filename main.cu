#include <iostream>
#include <fstream>

#include "gsieve.cuh"

using namespace std;

int main()
{
   // ifstream fin("../gsieve/sample/_samples96p");
   ifstream fin("sample/96a");
   GSieve gs("../gsieve/basis/basis102p", cin);

   gs.Start();
}