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
typedef double (*dev_fcn_ptr) (double); 


__device__ double dev_Gaussian (double xval) {
  double mean  = dev_params[dev_indices[1]]; // Not a typo
  double sigma = dev_params[dev_indices[2]]; 

  //printf("Gauss: %f %f %f %i %i\n", xval, mean, sigma, dev_indices[1], dev_indices[2]); 

  double ret = exp(-0.5*pow((xval - mean) / sigma, 2));
  ret /= sigma;
  ret /= sqrt(2*M_PI); 
  return ret;
}

__device__ double dev_BreitWigner (double xval) {
  double mean  = dev_params[dev_indices[1]];
  double width = dev_params[dev_indices[2]]; 

  double arg = xval - mean;
  double ret = width / (arg*arg + 0.25*width*width);
  ret /= (2*M_PI); // Normalise over -inf to inf 
  return ret; 
}

__device__ double dev_Polynomial (double xval) {
  // nP  c1  c2  c3  ...
  int numParams = dev_indices[0]; 

  double ret = 0; 
  for (int i = 0; i < numParams; ++i) {
    double coef  = dev_params[dev_indices[i + 1]];
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
      double coef  = dev_params[dev_indices[i + 1]];
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

__device__ dev_fcn_ptr ptr_to_Gaussian    = dev_Gaussian;
__device__ dev_fcn_ptr ptr_to_BreitWigner = dev_BreitWigner;
__device__ dev_fcn_ptr ptr_to_Polynomial  = dev_Polynomial; 
int host_fcnIdx = 0; 

// Note that 'unary_function' is ambiguous with STL class of same name. 
struct GeneralFcn : public thrust::unary_function<double, double> {

  GeneralFcn (unsigned int idx) 
    : fcnIdx(idx)
  {}

  __device__ double operator () (double xval) {
    dev_fcn_ptr theFunction;
    theFunction = reinterpret_cast<dev_fcn_ptr>(dev_fcn_table[fcnIdx]); 
    double pdfVal = (*theFunction)(xval); 
    return -2*log(pdfVal);
  }
  
private:
  unsigned int fcnIdx;
}; 

void fcn_glue (int& npar, double* deriv, double& fVal, double param[], int flag) {
  GeneralFcn functor(host_fcnIdx); 
  double initVal = 0; 
  cudaMemcpyToSymbol(dev_params, param, npar*sizeof(double)); 
  fVal = transform_reduce(dev_data->begin(),
			  dev_data->end(), 
			  functor, 
			  initVal, 
			  thrust::plus<double>()); // 'plus' also exists in STL. 
}

int main (int argc, char** argv) {
  if ((argc < 2) || (atoi(argv[1]) > 2) || (atoi(argv[1]) < 0)) {
    std::cout << "Usage: example3b N, where N is:\n"
	      << "  0: Gaussian fit\n"
	      << "  1: Breit-Wigner\n"
	      << "  2: Polynomial\n"; 
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
  cudaMemcpyFromSymbol(host_fcn_ptrs+2, ptr_to_Polynomial, sizeof(void*));
  cudaMemcpyToSymbol(dev_fcn_table, host_fcn_ptrs, 3*sizeof(void*)); 

  // Fit to Gaussian, Breit-Wigner, or polynomial. 
  host_fcnIdx = atoi(argv[1]); 
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
  return 0; 
}
