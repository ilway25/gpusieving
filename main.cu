#include <iostream>
#include <fstream>

#include "gsieve.cuh"

using namespace std;

int main()
{
   // ifstream fin("../gsieve/sample/_samples82p");
   GSieve gs("../gsieve/basis/basis82p", cin);

   gs.Start();
}