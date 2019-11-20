/**********************************************************************************************
** Terminology                                                                               **
***********************************************************************************************
** PGI:                                                                                      **
**   most commonly used OpenACC C/C++/Fortran compiler                                       **
** Device/accelerator:                                                                       **
**   the parallel processor we are trying to run on. examples would be a multicore CPU or GPU**
** Directive:                                                                                **
**   a command that OpenACC recognizes                                                       **
** Clause:                                                                                   **
**   additions to those commands to allow for more specific behavior                         **
***********************************************************************************************
** Syntax                                                                                    **
***********************************************************************************************
** #pragma acc <directive> <clauses>                                                         **
** Scoping rules:                                                                            **
**   some directives do not have any sort of scope (enter data, exit data, update). other    **
**   directives such as parallel, kernels, and data have to be scoped. the scope will either **
**   be the next immediate loop (for or while), or a region denoted by {}.                   **
** Implicit scopes:                                                                          **
**   when doing parallel loop or kernels loop, the loop it is applied to will be the scope.  **
**   these directives also create an implicit data region. for example...                    **
**   #pragma acc parallel loop present(...) functions the same as                            **
**   #pragma acc data present(...)                                                           **
**   #pragma acc parallel loop                                                               **
**********************************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <omp.h>
#include <openacc.h>

/**********************************************************************************************
** Matrix data structure                                                                     **
***********************************************************************************************
** enter data directive:                                                                     **
**   does device (GPU) allocation                                                            **
** exit data directive:                                                                      **
**   does device (GPU) deallocation                                                          **
** copyin clause:                                                                            **
**   specifies a host-to-device (CPU->GPU) data transfer                                     **
** create clause:                                                                            **
**   specifies that no other action other than allocation should occur                       **
** delete clause:                                                                            **
**   specifies that no other action other than deallocation should occur                     **
** update directive:                                                                         **
**   does a host-to-device or device-to-host data transfer                                   **
** self clause:                                                                              **
**   specifies that the data transfer is device-to-host                                      **
** device clause:                                                                            **
**   specifies that the data transfer is host-to-device                                      **
**********************************************************************************************/
struct matrix
{

  float * data;
  size_t nx, ny;

  matrix(int _nx, int _ny)
  {
    nx = _nx; ny = _ny;
    data = new float[_nx*_ny];
    #pragma acc enter data copyin(this)
    #pragma acc enter data create(data[:_nx*_ny])
  }

  ~matrix()
  {
    nx = 0; ny = 0;
    #pragma acc exit data delete(data)
    #pragma acc exit data delete(this)
    delete[] data;
  }

  float& at(int x, int y)
  {
    return data[x*ny + y];
  }

  void updateCPU()
  {
    #pragma acc update self(data[:nx*ny])
  }

  void updateGPU()
  {
    #pragma acc update device(data[:nx*ny])
  }

};

///////////////////////////////////////////////////////////////////////////////////////////////
// Vector data structure                                                                     //
///////////////////////////////////////////////////////////////////////////////////////////////
struct vector
{

  float * data;
  size_t n;

  vector(int _n)
  {
    n = _n;
    data = new float[_n];
    #pragma acc enter data copyin(this)
    #pragma acc enter data create(data[:_n])
  }

  ~vector()
  {
    n = 0;
    #pragma acc exit data delete(data)
    #pragma acc exit data delete(this)
    delete[] data;
  }

  float& at(int i)
  {
    return data[i];
  }

  void updateCPU()
  {
    #pragma acc update self(data[:n])
  }

  void updateGPU()
  {
    #pragma acc update device(data[:n])
  }

};


/**********************************************************************************************
** Dumb init functions                                                                       **
***********************************************************************************************
** parallel directive:                                                                       **
**   marks an area of the code that should be run on the accelerator (GPU).                  **
** loop directive:                                                                           **
**   marks a loop that should be parallelized on the accelerator (GPU).                      **
**   shorthand these two directives can be combined into one line as "parallel loop"         **
** collapse clause:                                                                          **
**   loop clause that combines nested loops, may increase data parallelism                   **
** present clause:                                                                           **
**   data clause that specifies data that is already allocated on the accelerator            **
**********************************************************************************************/
void init(matrix & mat, float val)
{
#pragma acc parallel loop collapse(2) \
 present(mat)
  for(int i = 0; i < mat.nx; i++)
    for(int j = 0; j < mat.ny; j++)
      mat.at(i, j) = val;
}

void init(vector & vec, float val)
{
#pragma acc parallel loop \
 present(vec)
  for(int i = 0; i < vec.n; i++)
    vec.at(i) = val;
}


/**********************************************************************************************
** Matrix-Vector muliply computation                                                         **
***********************************************************************************************
** reduction clause:                                                                         **
**   loop clause to allow parallel units to create a collective value                        **
** private clause:                                                                           **
**   loop clause that gives each parallel unit a private copy of a scalar or array           **
** gang clause:                                                                              **
**   loop clause that denotes coarse-grained parallelism. For multicore CPU this would be a  **
**   single CPU thread. For a GPU, it would be a block of GPU threads.                       **
** vector clause:                                                                            **
**   loop clause that denotes fine-grained parallelism. A good way to think of this is it    **
**   identifies SIMD operations. Some multicore CPUs and not take advantage of this, unless  **
**   it supports SIMD instructions. For GPUs, this would represent a single thread.          **
**********************************************************************************************/
void matvecmul(matrix & mat, vector & vec, vector & out)
{
  if(mat.ny != vec.n || mat.nx != out.n) {
    std::cerr << "matrix/vector dimensions incompatible" << std::endl;
    return;
  }

  int i, j;
  float sum;

#pragma acc parallel loop gang \
 present(mat, vec, out) \
 private(sum)
  for ( i = 0 ; i < mat.nx ; i++ ) {
    sum = 0.0f;
#pragma acc loop vector reduction(+:sum)
    for ( j = 0 ; j < mat.ny ; j++ ) {
      sum += mat.at(i,j)*vec.at(j);
    }
    out.at(i) = sum;
  }

}


///////////////////////////////////////////////////////////////////////////////////////////////
// Automated correctness checking                                                            //
///////////////////////////////////////////////////////////////////////////////////////////////
void check(matrix & mat, const char * name, const char * filename,
           const char * functionname, int linenum)
{
#ifdef DEBUG
  mat.updateCPU();
  pgi_compare(mat.data, "float", mat.nx*mat.ny, name, filename, functionname, linenum);
#endif
}

void check(vector & vec, const char * name, const char * filename,
           const char * functionname, int linenum)
{
#ifdef DEBUG
  vec.updateCPU();
  pgi_compare(vec.data, "float", vec.n, name, filename, functionname, linenum);
#endif
}


/**********************************************************************************************
** Main                                                                                      **
***********************************************************************************************
** Important steps:                                                                          **
**   Device-aware memory allocation                                                          **
**   Device computation                                                                      **
**   Correctness testing                                                                     **
**********************************************************************************************/
int main()
{

  matrix mat(128, 256);
  vector vec(256);
  vector out(128);

  init(mat, 1.0f);
  init(vec, 2.0f);

  matvecmul(mat, vec, out);

  check(mat, "mat", "OpenACCExample.cpp", "main", 1);
  check(vec, "vec", "OpenACCExample.cpp", "main", 2);
  check(out, "out", "OpenACCExample.cpp", "main", 3);

}

