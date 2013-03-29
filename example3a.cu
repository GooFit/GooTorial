#include "thrust/device_vector.h"
#include "thrust/host_vector.h"
#include "thrust/transform_reduce.h"
#include "rootstuff/TRandom.hh"
#include "rootstuff/TMinuit.hh" 

using namespace thrust; 

// Pointer to avoid crash on exit. 
device_vector<double>* dev_data;

// 512 functions should be enough for anyone.
__device__ void* dev_fcn_table[512];
typedef double (*dev_fcn_ptr) (double, double, double);


__device__ double dev_Gaussian (double xval, double mean, double sigma) {
  double ret = exp(-0.5*pow((xval - mean) / sigma, 2));
  ret /= sigma;
  ret /= sqrt(2*M_PI); 
  return ret;
}

__device__ double dev_BreitWigner (double xval, double mean, double width) {
  double arg = xval - mean;
  double ret = width / (arg*arg + 0.25*width*width);
  ret /= (2*M_PI); // Normalise over -inf to inf 
  return ret; 
}

__device__ dev_fcn_ptr ptr_to_Gaussian = dev_Gaussian;
__device__ dev_fcn_ptr ptr_to_BreitWigner = dev_BreitWigner;
int host_fcnIdx = 0; 

// Note that 'unary_function' is ambiguous with STL class of same name. 
struct GeneralFcn : public thrust::unary_function<double, double> {

  GeneralFcn (unsigned int idx, double p1, double p2)
    : fcnIdx(idx)
    , param1(p1)
    , param2(p2)
  {}

  __device__ double operator () (double xval) {
    dev_fcn_ptr theFunction;
    theFunction = reinterpret_cast<dev_fcn_ptr>(dev_fcn_table[fcnIdx]); 
    double pdfVal = (*theFunction)(xval, param1, param2); 
    return -2*log(pdfVal);
  }

private:
  unsigned int fcnIdx;
  double param1;
  double param2; 
}; 

void fcn_glue (int& npar, double* deriv, double& fVal, double param[], int flag) {
  GeneralFcn functor(host_fcnIdx, param[0], param[1]); 
  double initVal = 0; 
  fVal = transform_reduce(dev_data->begin(),
			  dev_data->end(), 
			  functor, 
			  initVal, 
			  thrust::plus<double>()); // 'plus' also exists in STL. 
}

int main (int argc, char** argv) {
  if ((argc < 2) || (atoi(argv[1]) > 1) || (atoi(argv[1]) < 0)) {
    std::cout << "Usage: example3a N, where N is either 0 (Gaussian fit) or 1 (Breit-Wigner).\n";
    return 1; 
  }

  // Generate random data
  TRandom donram(42); 
  host_vector<double> host_data;
  for (int i = 0; i < 10000; ++i) {
    host_data.push_back(donram.Gaus(0.0, 0.42));
  }
  // Move to device
  dev_data = new device_vector<double>(host_data); 
 
  // Initialise function table
  void* host_fcn_ptrs[512];
  cudaMemcpyFromSymbol(host_fcn_ptrs+0, ptr_to_Gaussian, sizeof(void*));
  cudaMemcpyFromSymbol(host_fcn_ptrs+1, ptr_to_BreitWigner, sizeof(void*));
  cudaMemcpyToSymbol(dev_fcn_table, host_fcn_ptrs, 2*sizeof(void*)); 

  // Fit to Gaussian or Breit-Wigner
  host_fcnIdx = atoi(argv[1]); 
  TMinuit minuit(2); 
  minuit.DefineParameter(0, "mean",  0.04, 0.01, -1.0, 1.0); 
  minuit.DefineParameter(1, "width", 0.50, 0.01, 0.0, 2.0); 
  minuit.SetFCN(fcn_glue);
  minuit.Migrad(); 

  // Free the device memory. 
  delete dev_data; 
  return 0; 
}
