#include "thrust/device_vector.h"
#include "thrust/host_vector.h"
#include "thrust/transform_reduce.h"
#include "rootstuff/TRandom.hh"
#include "rootstuff/TMinuit.hh" 
#include "cuPrintf.cu"

using namespace thrust; 

// Pointer to avoid crash on exit. 
device_vector<double>* dev_data;

// 512 functions should be enough for anyone.
__device__ void* dev_fcn_table[512];
__constant__ double dev_params[512]; 
__constant__ unsigned int dev_indices[512]; 
typedef double (*dev_fcn_ptr) (double, unsigned int); 


__device__ double dev_Gaussian (double xval, unsigned int pIdx) {
  double mean  = dev_params[dev_indices[pIdx + 1]]; // Not a typo
  double sigma = dev_params[dev_indices[pIdx + 2]]; 

  //printf("Gauss: %f %f %f %i %i\n", xval, mean, sigma, dev_indices[1], dev_indices[2]); 

  double ret = exp(-0.5*pow((xval - mean) / sigma, 2));
  ret /= sigma;
  ret /= sqrt(2*M_PI); 
  return ret;
}

__device__ double dev_BreitWigner (double xval, unsigned int pIdx) {
  double mean  = dev_params[dev_indices[pIdx + 1]];
  double width = dev_params[dev_indices[pIdx + 2]]; 

  double arg = xval - mean;
  double ret = width / (arg*arg + 0.25*width*width);
  ret /= (2*M_PI); // Normalise over -inf to inf 
  return ret; 
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

  // Not good to normalise from -inf to inf.
  // Avoid the problem for now by hardcoding 
  // integration limits. Note use of numerical
  // integration to avoid places where actual
  // polynomial goes negative - PDF is always
  // positive! 
  double integral = 0; 
  for (double xint = -5.0; xint < 5.0; xint += 0.01) {
    double curr = 0; 
    for (int i = 0; i < numParams; ++i) {
      double coef  = dev_params[dev_indices[pIdx + i + 1]];
      double power = pow(xint, i);
      curr        += coef*power;
    }
    if (curr < 0) continue;
    integral += 0.01*curr; 
  }

  //printf("Poly: %f %f %f\n", xval, ret, integral);

  ret /= integral;
  return max(ret, 1e-6); 
}

__device__ double dev_SumOfFunctions (double xval, unsigned int pIdx) {
  // EXERCISE: Implement this function so that it calculates the
  // weighted sum of an arbitrary number of other functions. 
  // It should assume that dev_indices stores (beginning at pIdx):
  // Total number of parameters for the sum (not for other functions)
  // Index (in dev_params) of the weight of the first function
  // Index (in dev_fcn_table) of the first function
  // Index (in dev_indices) where the parameters of the first function begin
  // (Repeat for second, third, n-1th functions)
  // Index of last function
  // Parameter index of last function.
  // 
  // Notice that the last function does not have a separate weight.
  // So for two functions the calculation is f_1 * P_1(x) + (1 - f_1) * P_2(x). 

  return 0; 
}

__device__ dev_fcn_ptr ptr_to_Gaussian        = dev_Gaussian;
__device__ dev_fcn_ptr ptr_to_BreitWigner     = dev_BreitWigner;
__device__ dev_fcn_ptr ptr_to_Polynomial      = dev_Polynomial; 
__device__ dev_fcn_ptr ptr_to_SumOfFunctions  = dev_SumOfFunctions; 
int host_fcnIdx = 0; 

// Note that 'unary_function' is ambiguous with STL class of same name. 
struct GeneralFcn : public thrust::unary_function<double, double> {

  GeneralFcn (unsigned int idx, unsigned int pid) 
    : fcnIdx(idx)
    , parIdx(pid)
  {}

  __device__ double operator () (double xval) {
    dev_fcn_ptr theFunction;
    theFunction = reinterpret_cast<dev_fcn_ptr>(dev_fcn_table[fcnIdx]); 
    double pdfVal = (*theFunction)(xval, parIdx); 
    return -2*log(pdfVal);
  }
  
private:
  unsigned int fcnIdx;
  unsigned int parIdx; 
}; 

void fcn_glue (int& npar, double* deriv, double& fVal, double param[], int flag) {
  GeneralFcn functor(host_fcnIdx, 0); 
  double initVal = 0; 
  cudaMemcpyToSymbol(dev_params, param, npar*sizeof(double)); 
  fVal = transform_reduce(dev_data->begin(),
			  dev_data->end(), 
			  functor, 
			  initVal, 
			  thrust::plus<double>());
}

int main (int argc, char** argv) {
  // Generate random data, two Gaussians with common mean. 
  TRandom donram(42); 
  host_vector<double> host_data;
  for (int i = 0; i < 100000; ++i) {
    double dieroll = donram.Uniform(); 
    if (dieroll < 0.2) host_data.push_back(donram.Gaus(0.0, 0.42));
    else host_data.push_back(donram.Gaus(0.0, 0.91));
  }
  // Move to device
  dev_data = new device_vector<double>(host_data); 
 
  // Initialise function table
  void* host_fcn_ptrs[512];
  cudaMemcpyFromSymbol(host_fcn_ptrs+0, ptr_to_Gaussian, sizeof(void*));
  cudaMemcpyFromSymbol(host_fcn_ptrs+1, ptr_to_BreitWigner, sizeof(void*));
  cudaMemcpyFromSymbol(host_fcn_ptrs+2, ptr_to_Polynomial, sizeof(void*));
  cudaMemcpyFromSymbol(host_fcn_ptrs+3, ptr_to_SumOfFunctions, sizeof(void*));
  cudaMemcpyToSymbol(dev_fcn_table, host_fcn_ptrs, 4*sizeof(void*)); 

  host_fcnIdx = 3; 
  unsigned int npars = 4;
  TMinuit minuit(npars); 

  // Set up Minuit to fit two Gaussians with common mean. 
  minuit.DefineParameter(0, "weight1",  0.30, 0.01, 0.0, 1.0); 
  minuit.DefineParameter(1, "mean",     0.30, 0.01, -10.0, 10.0); 
  minuit.DefineParameter(2, "sigma1",   0.20, 0.01, 0.01, 1.00); 
  minuit.DefineParameter(3, "sigma2",   0.70, 0.01, 0.20, 1.50); 

  // EXERCISE: Fill in the host_indices array!
  // It should contain a set of indices for the sum-function,
  // then for the first Gaussian, then for the second Gaussian. 
  // Don't forget to store the number of indices for each function. 
  // You probably want to write dev_SumOfFunctions first, so you
  // know the index structure it will use. 
  unsigned int host_indices[12]; 

  cudaMemcpyToSymbol(dev_indices, host_indices, 12*sizeof(unsigned int)); 
  minuit.SetFCN(fcn_glue);
  minuit.Migrad(); 

  // Free the device memory. 
  delete dev_data; 
  return 0; 
}
