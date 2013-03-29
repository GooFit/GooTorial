#include "TruthResolution_Aux.hh" 

__device__ fptype device_truth_resolution (fptype coshterm, fptype costerm, fptype sinhterm, fptype sinterm, 
					   fptype tau, fptype dtime, fptype xmixing, fptype ymixing, fptype /*sigma*/, 
					   fptype* /*p*/, unsigned int* /*indices*/) { 
  fptype ret = 0;
  dtime /= tau; 
  ret += coshterm*COSH(ymixing * dtime);
  ret += costerm*COS (xmixing * dtime);
  ret -= 2*sinhterm * SINH(ymixing * dtime);
  ret -= 2*sinterm * SIN (xmixing * dtime); // Notice sign difference wrt to Mikhail's code, because I have AB* and he has A*B. 
  ret *= EXP(-dtime); 

  //cuPrintf("device_truth_resolution %f %f %f %f %f\n", coshterm, costerm, sinhterm, sinterm, dtime); 
  return ret; 
}

__device__ device_resfunction_ptr ptr_to_truth = device_truth_resolution; 

TruthResolution::TruthResolution () 
  : MixingTimeResolution()
{
  cudaMemcpyFromSymbol((void**) &host_fcn_ptr, ptr_to_truth, sizeof(void*));
  initIndex(); 
}
TruthResolution::~TruthResolution () {} 

fptype TruthResolution::normalisation (fptype di1, fptype di2, fptype di3, fptype di4, fptype tau, fptype xmixing, fptype ymixing) const {
  fptype timeIntegralOne = tau / (1 - ymixing*ymixing); 
  fptype timeIntegralTwo = tau / (1 + xmixing*xmixing);
  fptype timeIntegralThr = ymixing * timeIntegralOne;
  fptype timeIntegralFou = xmixing * timeIntegralTwo;
       
  fptype ret = timeIntegralOne * (di1 + di2);
  ret       += timeIntegralTwo * (di1 - di2);
  ret       -= 2*timeIntegralThr * di3;
  ret       -= 2*timeIntegralFou * di4;

  return ret; 
}
