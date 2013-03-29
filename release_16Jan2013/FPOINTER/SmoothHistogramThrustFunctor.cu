#include "SmoothHistogramThrustFunctor.hh"

__constant__ fptype* dev_base_histograms[100]; // Multiple histograms for the case of multiple PDFs
__constant__ fptype* dev_smoothed_histograms[100]; 

__device__ int dev_powi (int base, int exp) {
  int ret = 1; 
  for (int i = 0; i < exp; ++i) ret *= base;
  return ret; 
}

__device__ fptype device_EvalHistogram (fptype* evt, fptype* p, unsigned int* indices) {
  // Structure is
  // nP smoothingIndex totalHistograms (limit1 step1 bins1) (limit2 step2 bins2) nO o1 o2
  // where limit and step are indices into functorConstants. 

  int numVars = indices[indices[0] + 1]; 
  int globalBinNumber = 0; 
  int previous = 1; 
  int myHistogramIndex = indices[2]; // 1 only used for smoothing

  for (int i = 0; i < numVars; ++i) { 
    int varIndex = indices[indices[0] + 2 + i]; 
    int lowerBoundIdx   = 3*(i+1);
    //if (gpuDebug & 1) printf("[%i, %i] Smoothed: %i %i %i\n", blockIdx.x, threadIdx.x, i, varIndex, indices[varIndex]); 
    fptype currVariable = evt[varIndex];
    fptype lowerBound   = functorConstants[indices[lowerBoundIdx + 0]];
    fptype step         = functorConstants[indices[lowerBoundIdx + 1]];

    currVariable -= lowerBound;
    currVariable /= step; 
    //if (gpuDebug & 1) printf("[%i, %i] Smoothed: %i %i %f %f %f %f\n", blockIdx.x, threadIdx.x, i, varIndex, currVariable, lowerBound, step, evt[varIndex]); 

    int localBinNumber  = (int) FLOOR(currVariable); 
    globalBinNumber    += previous * localBinNumber; 
    previous           *= indices[lowerBoundIdx + 2];
  }

  fptype* myHistogram = dev_smoothed_histograms[myHistogramIndex];
  fptype ret = myHistogram[globalBinNumber];

  //if ((gpuDebug & 1) && (paramIndices + debugParamIndex == indices)) printf("Smoothed: %f %f %f %i %f\n", evt[0], evt[1], myHistogram[globalBinNumber], globalBinNumber, dev_base_histograms[myHistogramIndex][globalBinNumber]);
  //if (gpuDebug & 1) printf("Smoothed: %f %f %f %i %f\n", evt[0], evt[1], myHistogram[globalBinNumber], globalBinNumber, dev_base_histograms[myHistogramIndex][globalBinNumber]);
  //if (gpuDebug & 1) printf("Smoothed: %f %f %f %i %f %f\n", evt[0], evt[1], ret, globalBinNumber, dev_base_histograms[myHistogramIndex][globalBinNumber], p[indices[1]]);
  return ret; 
}

struct Smoother { 
  int parameters;

  __device__ fptype operator () (int globalBin) {
    unsigned int* indices = paramIndices + parameters; 
    int numVars = indices[indices[0] + 1]; 
    fptype smoothing = cudaArray[indices[1]];
    int histIndex = indices[2]; 
    fptype* myHistogram = dev_base_histograms[histIndex];
    fptype centralValue = myHistogram[globalBin]; 

    fptype otherBinsTotal = 0; 
    int numSurroundingBins = 0; 
    int otherBins = dev_powi(3, numVars); 
    for (int i = 0; i < otherBins; ++i) {
      int currBin = globalBin; 
      int localPrevious = 1; 
      int trackingBin = globalBin; 
      bool offSomeAxis = false; 
      for (int v = 0; v < numVars; ++v) {
	//int lowerBoundIdx   = 3*(i+1);
	//int localNumBins = indices[6 + v*4];
	int localNumBins = indices[3*(v+1) + 2];
	int offset = ((i / dev_powi(3, v)) % 3) - 1; 
	
	currBin += offset * localPrevious; 
	localPrevious *= localNumBins; 
	
	int currVarBin = trackingBin % localNumBins; 
	trackingBin /= localNumBins; 
	if (currVarBin + offset < 0) offSomeAxis = true;
	if (currVarBin + offset >= localNumBins) offSomeAxis = true;
      }
      
      if (currBin == globalBin) continue; 
      if (offSomeAxis) continue; // Out of bounds 
      numSurroundingBins++; 
      
      otherBinsTotal += myHistogram[currBin]; 
    }
    
    centralValue += otherBinsTotal*smoothing;
    centralValue /= (1 + numSurroundingBins * smoothing);

    //if (5000 == globalBin) printf("Smoothing: %f %f %f %f\n", myHistogram[globalBin], otherBinsTotal, smoothing, centralValue); 
    return centralValue; 
  }
};

__device__ device_function_ptr ptr_to_EvalHistogram = device_EvalHistogram; 

__host__ SmoothHistogramThrustFunctor::SmoothHistogramThrustFunctor (std::string n, BinnedDataSet* x, Variable* smoothing) 
  : ThrustPdfFunctor(0, n) 
{
  int numVars = x->numVariables(); 
  int numConstants = 2*numVars;
  registerConstants(numConstants);
  static unsigned int totalHistograms = 0; 
  host_constants = new fptype[numConstants]; 
  totalEvents = 0; 

  std::vector<unsigned int> pindices;
  pindices.push_back(registerParameter(smoothing));
  pindices.push_back(totalHistograms); 

  int varIndex = 0; 
  for (varConstIt var = x->varsBegin(); var != x->varsEnd(); ++var) {
    registerObservable(*var);
    //pindices.push_back((*var)->index); 
    pindices.push_back(cIndex + 2*varIndex + 0);
    pindices.push_back(cIndex + 2*varIndex + 1);
    pindices.push_back((*var)->numbins);

    host_constants[2*varIndex + 0] = (*var)->lowerlimit; // NB, do not put cIndex here, it is accounted for by the offset in cudaMemcpyToSymbol below. 
    host_constants[2*varIndex + 1] = ((*var)->upperlimit - (*var)->lowerlimit) / (*var)->numbins; 
    varIndex++; 
  }

  unsigned int numbins = x->getNumBins(); 
  thrust::host_vector<fptype> host_histogram; 
  for (unsigned int i = 0; i < numbins; ++i) {
    fptype curr = x->getBinContent(i);
    host_histogram.push_back(curr);
    totalEvents += curr; 
  }
  cudaMemcpyToSymbol(functorConstants, host_constants, numConstants*sizeof(fptype), cIndex*sizeof(fptype), cudaMemcpyHostToDevice); 

  dev_base_histogram = new thrust::device_vector<fptype>(host_histogram);  
  dev_smoothed_histogram = new thrust::device_vector<fptype>(host_histogram);  
  static fptype* dev_address[1];
  dev_address[0] = (&((*dev_base_histogram)[0])).get();
  cudaMemcpyToSymbol(dev_base_histograms, dev_address, sizeof(fptype*), totalHistograms*sizeof(fptype), cudaMemcpyHostToDevice); 
  dev_address[0] = (&((*dev_smoothed_histogram)[0])).get();
  cudaMemcpyToSymbol(dev_smoothed_histograms, dev_address, sizeof(fptype*), totalHistograms*sizeof(fptype), cudaMemcpyHostToDevice); 
  cudaMemcpyFromSymbol((void**) &host_fcn_ptr, ptr_to_EvalHistogram, sizeof(void*));
  initialise(pindices); 

  totalHistograms++; 
}

__host__ fptype SmoothHistogramThrustFunctor::normalise () const {
  Smoother smoother;
  smoother.parameters = parameters;

  thrust::counting_iterator<int> binIndex(0); 
  thrust::transform(binIndex, 
		    binIndex + dev_base_histogram->size(), 
		    dev_smoothed_histogram->begin(),
		    smoother);

  //return totalEvents; 
  fptype ret = thrust::reduce(dev_smoothed_histogram->begin(), dev_smoothed_histogram->end()); 

  for (unsigned int varIndex = 0; varIndex < observables.size(); ++varIndex) {
    ret *= host_constants[2*varIndex + 1]; // Bin size cached by constructor. 
  }

  //if (cpuDebug & 1) std::cout << "Normalising " << getName() << " " << host_params[host_indices[parameters + 1]] << " " << ret << std::endl; 
  host_normalisation[parameters] = 1.0/ret; 
  return ret; 
}
