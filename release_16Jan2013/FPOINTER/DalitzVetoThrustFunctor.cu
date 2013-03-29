#include "DalitzVetoThrustFunctor.hh"
#include "TddpHelperFunctions.hh" 

__device__ fptype device_DalitzVeto (fptype* evt, fptype* p, unsigned int* indices) {
  fptype x         = evt[indices[2 + indices[0] + 0]]; 
  fptype y         = evt[indices[2 + indices[0] + 1]]; 

  fptype motherM   = p[indices[1]];
  fptype d1m       = p[indices[2]];
  fptype d2m       = p[indices[3]];
  fptype d3m       = p[indices[4]];

  fptype massSum   = motherM*motherM + d1m*d1m + d2m*d2m + d3m*d3m;
  fptype z         = massSum - x - y;

  fptype ret = inDalitz(x, y, motherM, d1m, d2m, d3m) ? 1.0 : 0.0; 
  unsigned int numVetos = indices[5];
  for (int i = 0; i < numVetos; ++i) {
    unsigned int varIndex =   indices[6 + i*3 + 0];
    fptype minimum        = p[indices[6 + i*3 + 1]];
    fptype maximum        = p[indices[6 + i*3 + 2]];
    fptype currDalitzVar = (PAIR_12 == varIndex ? x : PAIR_13 == varIndex ? y : z);

    ret *= ((currDalitzVar < maximum) && (currDalitzVar > minimum)) ? 0.0 : 1.0;
  }

  return ret; 
}

__device__ device_function_ptr ptr_to_DalitzVeto = device_DalitzVeto;



__host__ DalitzVetoThrustFunctor::DalitzVetoThrustFunctor (std::string n, Variable* _x, Variable* _y, Variable* motherM, Variable* d1m, Variable* d2m, Variable* d3m, vector<VetoInfo*> vetos) 
  : ThrustPdfFunctor(0, n) 
{
  registerObservable(_x);
  registerObservable(_y);

  std::vector<unsigned int> pindices;
  pindices.push_back(registerParameter(motherM));
  pindices.push_back(registerParameter(d1m));
  pindices.push_back(registerParameter(d2m));
  pindices.push_back(registerParameter(d3m));

  pindices.push_back(vetos.size()); 
  for (vector<VetoInfo*>::iterator v = vetos.begin(); v != vetos.end(); ++v) {
    pindices.push_back((*v)->cyclic_index);
    pindices.push_back(registerParameter((*v)->minimum));
    pindices.push_back(registerParameter((*v)->maximum));
  }


  cudaMemcpyFromSymbol((void**) &host_fcn_ptr, ptr_to_DalitzVeto, sizeof(void*));
  initialise(pindices); 
}
