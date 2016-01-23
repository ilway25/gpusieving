#include <iostream>
#include <fstream>

#include "gsieve.cuh"

using namespace std;

int main()
{
   ifstream fin("../gsieve/sample/_samples96p");
   GSieve gs("../gsieve/basis/basis96p", cin);

   gs.Start();
}