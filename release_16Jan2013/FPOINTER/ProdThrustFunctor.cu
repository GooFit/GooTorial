#include "ProdThrustFunctor.hh"

__device__ fptype device_ProdPdfs (fptype* evt, fptype* p, unsigned int* indices) { 
  // Index structure is nP | F1 P1 | F2 P2 | ...
  // where nP is number of parameters, Fs are function indices, and Ps are parameter indices

  int numParams = indices[0]; 
  fptype ret = 1;

  for (int i = 1; i < numParams; i += 2) {
    int fcnIdx = indices[i + 0]; 
    int parIdx = indices[i + 1]; 

    fptype curr = (*(reinterpret_cast<device_function_ptr>(device_function_table[fcnIdx])))(evt, p, paramIndices + parIdx);
    curr *= normalisationFactors[parIdx]; 
    ret *= curr;

    //if ((0 == threadIdx.x) && (0 == blockIdx.x) && (gpuDebug & 1) && (paramIndices + debugParamIndex == indices))
    //if ((1 > (int) floor(0.5 + evt[8])) && (gpuDebug & 1) && (paramIndices + debugParamIndex == indices))
    //if (0.0001 < ret) 
    //if ((gpuDebug & 1) && (isnan(curr)))
    //if (gpuDebug & 1)
    //if ((gpuDebug & 1) && (0 == threadIdx.x))
    //printf("device_Prod: (%f %f %f %f %f) %f %f %f %i %i %i\n", evt[0], evt[1], evt[2], evt[3], evt[4], curr, ret, normalisationFactors[parIdx], i, parIdx, numParams);
    //printf("(%i, %i) device_Prod: (%f %f %f %f) %f %f %f %i\n", blockIdx.x, threadIdx.x, evt[0], evt[8], evt[6], evt[7], curr, ret, normalisationFactors[parIdx], i);
    //printf("(%i, %i) device_Prod: (%f %f) %f %f %f %i\n", blockIdx.x, threadIdx.x, evt[0], evt[1], curr, ret, normalisationFactors[parIdx], i);

  }

  return ret; 
}

__device__ device_function_ptr ptr_to_ProdPdfs = device_ProdPdfs; 

ProdThrustFunctor::ProdThrustFunctor (std::string n, std::vector<FunctorBase*> comps) 
  : ThrustPdfFunctor(0, n) 
  , varOverlaps(false) 
{
  std::vector<unsigned int> pindices;

  for (std::vector<FunctorBase*>::iterator p = comps.begin(); p != comps.end(); ++p) {
    assert(*p);
    components.push_back(*p); 
  }

  getObservables(observables); // Gathers from components

  FunctorBase::obsCont observableCheck; // Use to check for overlap in observables

  // Indices stores (function index)(function parameter index)(variable index) for each component.
  for (std::vector<FunctorBase*>::iterator p = comps.begin(); p != comps.end(); ++p) {
    pindices.push_back((*p)->getFunctionIndex());
    pindices.push_back((*p)->getParameterIndex());

    if (varOverlaps) continue; // Only need to establish this once. 
    FunctorBase::obsCont currObses;
    (*p)->getObservables(currObses); 
    for (FunctorBase::obsIter o = currObses.begin(); o != currObses.end(); ++o) {
      if (find(observableCheck.begin(), observableCheck.end(), (*o)) == observableCheck.end()) continue; 
      varOverlaps = true;
      break;
    }
    (*p)->getObservables(observableCheck); 
  }
  
  if (varOverlaps) { // Check for components forcing separate normalisation
    for (std::vector<FunctorBase*>::iterator p = comps.begin(); p != comps.end(); ++p) {
      if ((*p)->getSpecialMask() & FunctorBase::ForceSeparateNorm) varOverlaps = false; 
    }
  }

  cudaMemcpyFromSymbol((void**) &host_fcn_ptr, ptr_to_ProdPdfs, sizeof(void*));
  initialise(pindices); 
} 

__host__ fptype ProdThrustFunctor::normalise () const {
  
  if (varOverlaps) {
    // Two or more components share an observable and cannot be separately
    // normalised, since \int A*B dx does not equal int A dx * int B dx. 
    recursiveSetNormalisation(fptype(1.0)); 
    cudaMemcpyToSymbol(normalisationFactors, host_normalisation, totalParams*sizeof(fptype), 0, cudaMemcpyHostToDevice); 
    
    // Normalise numerically.
    //std::cout << "Numerical normalisation of " << getName() << " due to varOverlaps.\n"; 
    fptype ret = ThrustPdfFunctor::normalise();
    //if (cpuDebug & 1) 
    //std::cout << "ProdThrustFunctor " << getName() << " has normalisation " << ret << " " << host_callnumber << std::endl; 
    return ret;
  }
  
  // Normalise components individually 
  for (std::vector<FunctorBase*>::const_iterator c = components.begin(); c != components.end(); ++c) {
    (*c)->normalise(); 
  }
  host_normalisation[parameters] = 1; 
  cudaMemcpyToSymbol(normalisationFactors, host_normalisation, totalParams*sizeof(fptype), 0, cudaMemcpyHostToDevice); 
  
  return 1.0; 
}
