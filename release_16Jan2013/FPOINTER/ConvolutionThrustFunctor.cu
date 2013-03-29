#include "ConvolutionThrustFunctor.hh"

  int totalConvolutions = 0; 
#ifdef OMP_ON
#pragma omp threadprivate(totalConvolutions)
#endif

// Need multiple working spaces for the case of several convolutions in one PDF. 
__constant__ fptype* dev_modWorkSpace[100];
__constant__ fptype* dev_resWorkSpace[100]; 

// Number which transforms model range (x1, x2) into resolution range (x1 - maxX, x2 - minX).
// It is equal to the maximum possible value of x0, ie maxX, in bins. 
__constant__ int modelOffset = 0; 

__device__ fptype device_ConvolvePdfs (fptype* evt, fptype* p, unsigned int* indices) { 
  fptype ret     = 0; 
  fptype loBound = functorConstants[indices[5]+0];
  fptype hiBound = functorConstants[indices[5]+1];
  fptype step    = functorConstants[indices[5]+2];
  fptype x0      = evt[indices[2 + indices[0]]]; 
  int workSpaceIndex = indices[6]; 

  int numbins = (int) FLOOR((hiBound - loBound) / step); 

  fptype lowerBoundOffset = loBound / step;
  lowerBoundOffset -= FLOOR(lowerBoundOffset); 
  int offsetInBins = (int) FLOOR(x0 / step - lowerBoundOffset); 

  // integral M(x) * R(x - x0) dx
  for (int i = 0; i < numbins; ++i) {
    fptype model = dev_modWorkSpace[workSpaceIndex][i]; 
    fptype resol = dev_resWorkSpace[workSpaceIndex][i + modelOffset - offsetInBins]; 
    ret += model*resol;
  }

  ret *= normalisationFactors[indices[2]]; 
  ret *= normalisationFactors[indices[4]]; 

  return ret; 
}

__device__ device_function_ptr ptr_to_ConvolvePdfs = device_ConvolvePdfs; 

ConvolutionThrustFunctor::ConvolutionThrustFunctor (std::string n,
						    Variable* x, 
						    ThrustPdfFunctor* m, 
						    ThrustPdfFunctor* r) 
  : ThrustPdfFunctor(x, n)
  , model(m)
  , resolution(r)
  , host_iConsts(0)
  , modelWorkSpace(0)
  , resolWorkSpace(0)
  , workSpaceIndex(0)
{
  components.push_back(model);
  components.push_back(resolution);

//  static int totalConvolutions = 0; 

  // Indices stores (function index)(parameter index) doublet for model and resolution function. 
  std::vector<unsigned int> paramIndices;
  paramIndices.push_back(model->getFunctionIndex());
  paramIndices.push_back(model->getParameterIndex()); 
  paramIndices.push_back(resolution->getFunctionIndex());
  paramIndices.push_back(resolution->getParameterIndex()); 
  paramIndices.push_back(registerConstants(3));
  paramIndices.push_back(workSpaceIndex = totalConvolutions++); 
  
  cudaMemcpyFromSymbol((void**) &host_fcn_ptr, ptr_to_ConvolvePdfs, sizeof(void*));
  initialise(paramIndices);
  setIntegrationConstants(-10, 10, 0.01);
} 

__host__ void ConvolutionThrustFunctor::setIntegrationConstants (fptype lo, fptype hi, fptype step) {
  if (!host_iConsts) {
    host_iConsts = new fptype[6]; 
    cudaMalloc((void**) &dev_iConsts, 6*sizeof(fptype)); 
  }
  host_iConsts[0] = lo;
  host_iConsts[1] = hi;
  host_iConsts[2] = step;
  cudaMemcpyToSymbol(functorConstants, host_iConsts, 3*sizeof(fptype), cIndex*sizeof(fptype), cudaMemcpyHostToDevice); 
  if (modelWorkSpace) {
    delete modelWorkSpace;
    delete resolWorkSpace;
  }

  int numbins = (int) floor((host_iConsts[1] - host_iConsts[0]) / step); 
  // Different format for integration range! 
  modelWorkSpace = new thrust::device_vector<fptype>(numbins);

  // We will do integral from x1 to x2 of M(x)*R(x - x0) dx.
  // So we need to cache the values of M from x1 to x2, which is given
  // by the integration range. But R must be cached from x1-maxX to
  // x2-minX, and the min and max are given by the dependent variable. 
  // However, the step must be the same as for the model, or the binning
  // will get out of sync. 
  Variable* dependent = *(observables.begin()); 

  host_iConsts[2] = numbins; 
  host_iConsts[3] = (host_iConsts[0] - dependent->upperlimit);
  host_iConsts[4] = (host_iConsts[1] - dependent->lowerlimit);

  numbins = (int) floor((host_iConsts[4] - host_iConsts[3]) / step); 
  host_iConsts[5] = numbins; 
  cudaMemcpy(dev_iConsts, host_iConsts, 6*sizeof(fptype), cudaMemcpyHostToDevice); 
  resolWorkSpace = new thrust::device_vector<fptype>(numbins);

  // NB, this could potentially be a problem with multiple convolutions. 
  int offset = dependent->upperlimit / step; 
  cudaMemcpyToSymbol(modelOffset, &offset, sizeof(int), 0, cudaMemcpyHostToDevice); 

  fptype* dev_address[1];
  dev_address[0] = (&((*modelWorkSpace)[0])).get();
  cudaMemcpyToSymbol(dev_modWorkSpace, dev_address, sizeof(fptype*), workSpaceIndex*sizeof(fptype), cudaMemcpyHostToDevice); 
  dev_address[0] = (&((*resolWorkSpace)[0])).get();
  cudaMemcpyToSymbol(dev_resWorkSpace, dev_address, sizeof(fptype*), workSpaceIndex*sizeof(fptype), cudaMemcpyHostToDevice); 
}

__host__ fptype ConvolutionThrustFunctor::normalise () const {
  // First set normalisation factors to one so we can evaluate convolution without getting zeroes
  recursiveSetNormalisation(fptype(1.0)); 
  cudaMemcpyToSymbol(normalisationFactors, host_normalisation, totalParams*sizeof(fptype), 0, cudaMemcpyHostToDevice); 

  // Next recalculate functions at each point, in preparation for convolution integral
  thrust::constant_iterator<fptype*> arrayAddress(dev_iConsts); 
  thrust::constant_iterator<int> eventSize(1);
  thrust::counting_iterator<int> binIndex(0); 

  MetricTaker modalor(model, getMetricPointer("ptr_to_Eval")); 
  thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(binIndex, eventSize, arrayAddress)),
		    thrust::make_zip_iterator(thrust::make_tuple(binIndex + modelWorkSpace->size(), eventSize, arrayAddress)),
		    modelWorkSpace->begin(), 
		    modalor);

  thrust::constant_iterator<fptype*> arrayAddress2(dev_iConsts + 3); 
  MetricTaker resalor(resolution, getMetricPointer("ptr_to_Eval")); 
  thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(binIndex, eventSize, arrayAddress2)),
		    thrust::make_zip_iterator(thrust::make_tuple(binIndex + resolWorkSpace->size(), eventSize, arrayAddress2)),
		    resolWorkSpace->begin(), 
		    resalor);

  // Then return usual integral
  fptype ret = ThrustPdfFunctor::normalise();
  return ret; 
}



