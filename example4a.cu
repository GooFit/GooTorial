#include "thrust/device_vector.h"
#include "thrust/host_vector.h"
#include "thrust/transform_reduce.h"
#include "rootstuff/TRandom.hh"
#include "rootstuff/TMinuit.hh" 
#include "cuPrintf.cu"
#include <vector>
#include <utility> 

using namespace thrust; 

// Pointer to avoid crash on exit. 
device_vector<double>* dev_data;
device_vector<double>* dev_norm_data; 

// 512 functions should be enough for anyone.
__device__ void* dev_fcn_table[512];
__constant__ double dev_params[512]; 
__constant__ unsigned int dev_indices[512]; 
__constant__ double dev_norm_integrals[512]; 
double host_norm_integrals[512]; 
typedef double (*dev_fcn_ptr) (double, unsigned int); 


__device__ double dev_Gaussian (double xval, unsigned int pIdx) {
  double mean  = dev_params[dev_indices[pIdx + 1]]; 
  double sigma = dev_params[dev_indices[pIdx + 2]]; 
  return exp(-0.5*pow((xval - mean) / sigma, 2)); // External normalisation
}

__device__ double dev_BreitWigner (double xval, unsigned int pIdx) {
  double mean  = dev_params[dev_indices[pIdx + 1]];
  double width = dev_params[dev_indices[pIdx + 2]]; 
  double arg = xval - mean;
  return 1.0 / (arg*arg + 0.25*width*width);
}

__device__ double dev_Polynomial (double xval, unsigned int pIdx) {
  // nP  c1  c2  c3  ...
  int numParams = dev_indices[pIdx]; 

  double ret = 0; 
  for (int i = 0; i < numParams; ++i) {
    double coef  = dev_params[dev_indices[pIdx + i + 1]];
    double power = pow(xval, i);
    ret         += coef*power; 
  }

  return max(ret, 1e-6); 
}

__device__ dev_fcn_ptr ptr_to_Gaussian    = dev_Gaussian;
__device__ dev_fcn_ptr ptr_to_BreitWigner = dev_BreitWigner;
__device__ dev_fcn_ptr ptr_to_Polynomial  = dev_Polynomial; 
int host_fcnIdx = 0; 

__device__ double callFunction (double xval, unsigned int fcnIdx, unsigned int parIdx) {
  dev_fcn_ptr theFunction;
  theFunction = reinterpret_cast<dev_fcn_ptr>(dev_fcn_table[fcnIdx]); 
  double pdfVal = (*theFunction)(xval, parIdx); 
  pdfVal *= dev_norm_integrals[parIdx]; 
  return pdfVal; 
}

struct GeneralFcn : public thrust::unary_function<double, double> {
  GeneralFcn (unsigned int idx, unsigned int pid, bool tl = true) 
    : fcnIdx(idx)
    , parIdx(pid)
    , takeLog(tl)
  {}

  __device__ double operator () (double xval) {
    double pdfVal = callFunction(xval, fcnIdx, parIdx); 
    if (takeLog) return -2*log(pdfVal);
    else return pdfVal; 
  }
  
private:
  unsigned int fcnIdx;
  unsigned int parIdx; 
  bool takeLog; 
}; 

std::vector<std::pair<unsigned int, unsigned int> > fcns_to_normalise; 

void fcn_glue (int& npar, double* deriv, double& fVal, double param[], int flag) {
  cudaMemcpyToSymbol(dev_params, param, npar*sizeof(double)); 
  double initVal = 0; 

  for (int i = 0; i < 512; ++i) host_norm_integrals[i] = 1.0; 
  cudaMemcpyToSymbol(dev_norm_integrals, host_norm_integrals, 512*sizeof(double));

  for (unsigned int i = 0; i < fcns_to_normalise.size(); ++i) {
    unsigned int fcnIdx = fcns_to_normalise[i].first;
    unsigned int parIdx = fcns_to_normalise[i].second;
    GeneralFcn normaliser(fcnIdx, parIdx, false);
    initVal = 0; 
    double integral = thrust::transform_reduce(dev_norm_data->begin(), 
					       dev_norm_data->end(), 
					       normaliser, 
					       initVal, 
					       thrust::plus<double>());
    integral *= 0.001; // (Cheating by hardcoding step size!) 
    //std::cout << "Normalisation: " << integral << std::endl; 
    host_norm_integrals[parIdx] = 1.0 / integral; 
  }
  cudaMemcpyToSymbol(dev_norm_integrals, host_norm_integrals, 512*sizeof(double));
  
  initVal = 0; 
  GeneralFcn functor(host_fcnIdx, 0); 
  fVal = transform_reduce(dev_data->begin(),
			  dev_data->end(), 
			  functor, 
			  initVal, 
			  thrust::plus<double>()); 
  //std::cout << fVal << " " << param[0] << " " << param[1] << " " << param[2] << std::endl;
}

int main (int argc, char** argv) {
  // Generate random data
  TRandom donram(42); 
  host_vector<double> host_data;
  for (int i = 0; i < 10000; ++i) {
    host_data.push_back(donram.Gaus(0.0, 0.42));
  }
  // Move to device
  dev_data = new device_vector<double>(host_data); 

  // Create normalisation-integral points
  host_data.clear(); 
  for (double xval = -5.0; xval < 5.0; xval += 0.001) {
    host_data.push_back(xval); 
  }
  dev_norm_data = new device_vector<double>(host_data); 

  // Initialise function table
  void* host_fcn_ptrs[512];
  cudaMemcpyFromSymbol(host_fcn_ptrs+0, ptr_to_Gaussian, sizeof(void*));
  cudaMemcpyFromSymbol(host_fcn_ptrs+1, ptr_to_BreitWigner, sizeof(void*));
  cudaMemcpyFromSymbol(host_fcn_ptrs+2, ptr_to_Polynomial, sizeof(void*));
  cudaMemcpyToSymbol(dev_fcn_table, host_fcn_ptrs, 3*sizeof(void*)); 

  // Fit to Gaussian, Breit-Wigner, or polynomial. 
  host_fcnIdx = atoi(argv[1]); 
  fcns_to_normalise.push_back(std::pair<unsigned int, unsigned int>(host_fcnIdx, 0)); 
  unsigned int npars = (host_fcnIdx == 2 ? 3 : 2);
  TMinuit minuit(npars); 

  if (3 == npars) {
    minuit.DefineParameter(0, "cons",  1.00, 0.01, -10.0, 10.0); 
    minuit.DefineParameter(1, "line",  0.00, 0.01, -10.0, 10.0); 
    minuit.DefineParameter(2, "quad", -1.00, 0.01, -10.0, 10.0); 
  }
  else {
    minuit.DefineParameter(0, "mean",  0.04, 0.01, -1.0, 1.0); 
    minuit.DefineParameter(1, "width", 0.50, 0.01, 0.0, 2.0); 
  }

  unsigned int host_indices[10]; 
  host_indices[0] = npars; 
  for (unsigned int i = 0; i < npars; ++i) host_indices[i+1] = i; 
  cudaMemcpyToSymbol(dev_indices, host_indices, (1+npars)*sizeof(unsigned int)); 
  minuit.SetFCN(fcn_glue);
  minuit.Migrad(); 



  // Free the device memory. 
  delete dev_data; 
  delete dev_norm_data; 
  return 0; 
}
