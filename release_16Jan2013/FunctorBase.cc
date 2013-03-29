#include "GlobalCudaDefines.hh"
#include "FunctorBase.hh" 

fptype* cudaDataArray;
fptype host_normalisation[maxParams];
fptype host_params[maxParams];
unsigned int host_indices[maxParams]; 
int host_callnumber = 0; 
int totalParams = 0; 
int totalConstants = 1; // First constant is reserved for number of events. 

#ifdef OMP_ON
typedef std::map<Variable*, std::set<FunctorBase*> > tRegType;
tRegType variableRegistry[MAX_THREADS]; 
#pragma omp threadprivate (cudaDataArray, totalParams, totalConstants)
#pragma omp threadprivate (host_normalisation, host_params, host_indices)
#pragma omp threadprivate (host_callnumber)
#else
std::map<Variable*, std::set<FunctorBase*> > variableRegistry; 
#endif

FunctorBase::FunctorBase (Variable* x, std::string n) 
  : name(n)
  , numEvents(0)
  , numEntries(0)
  , normRanges(0)
  , fitControl(0) 
  , integrationBins(-1) 
  , specialMask(0)
  , cachedParams(0)
{
  if (x) registerObservable(x);
}


__host__ void FunctorBase::recursiveSetNormalisation (fptype norm) const {
  host_normalisation[parameters] = norm;
  for (unsigned int i = 0; i < components.size(); ++i) {
    components[i]->recursiveSetNormalisation(norm); 
  }
}

__host__ unsigned int FunctorBase::registerParameter (Variable* var) {
  if (!var) {
    std::cout << "Error: Attempt to register null Variable with " 
	      << getName()
	      << ", aborting.\n";
    assert(var);
    exit(1); 
  }
  
  if (std::find(parameterList.begin(), parameterList.end(), var) != parameterList.end()) return (unsigned int) var->getIndex(); 
  
  parameterList.insert(var); 
#ifdef OMP_ON
  int tid = omp_get_thread_num();
  variableRegistry[tid][var].insert(this); 
#else
  variableRegistry[var].insert(this); 
#endif
  if (0 > var->getIndex()) {
    unsigned int unusedIndex = 0;
    while (true) {
      bool canUse = true;
#ifdef OMP_ON
      for (std::map<Variable*, std::set<FunctorBase*> >::iterator p = variableRegistry[tid].begin(); p != variableRegistry[tid].end(); ++p) {
#else
      for (std::map<Variable*, std::set<FunctorBase*> >::iterator p = variableRegistry.begin(); p != variableRegistry.end(); ++p) {
#endif
	if (unusedIndex != (*p).first->index) continue;
	canUse = false;
	break; 
      }
      if (canUse) break; 
      unusedIndex++; 
    }
      
    var->index = unusedIndex; 
  }
  return (unsigned int) var->getIndex(); 
}

__host__ void FunctorBase::unregisterParameter (Variable* var) {
  if (!var) return; 
  parameterList.erase(var); 
#ifdef OMP_ON
  int tid = omp_get_thread_num();
  variableRegistry[tid][var].erase(this);
  if (0 == variableRegistry[tid][var].size()) var->index = -1; 
#else
  variableRegistry[var].erase(this);
  if (0 == variableRegistry[var].size()) var->index = -1; 
#endif
  for (unsigned int i = 0; i < components.size(); ++i) {
    components[i]->unregisterParameter(var); 
  }
}

__host__ void FunctorBase::getParameters (std::set<Variable*>& ret) const { 
  for (std::set<Variable*>::iterator p = parameterList.begin(); p != parameterList.end(); ++p) ret.insert(*p); 
  for (unsigned int i = 0; i < components.size(); ++i) {
    components[i]->getParameters(ret); 
  }
}

__host__ void FunctorBase::getObservables (std::vector<Variable*>& ret) const { 
  for (obsConstIter p = obsCBegin(); p != obsCEnd(); ++p) {
    if (std::find(ret.begin(), ret.end(), *p) != ret.end()) continue; 
    ret.push_back(*p); 
  }
  for (unsigned int i = 0; i < components.size(); ++i) {
    components[i]->getObservables(ret); 
  }
}

__host__ unsigned int FunctorBase::registerConstants (unsigned int amount) {
  assert(totalConstants + amount < maxParams);
  cIndex = totalConstants;
  totalConstants += amount; 
  return cIndex; 
}

void FunctorBase::registerObservable (Variable* obs) {
  if (!obs) return;
  if (find(observables.begin(), observables.end(), obs) != observables.end()) return; 
  observables.push_back(obs);
}

__host__ void FunctorBase::setIntegrationFineness (int i) {
  integrationBins = i;
  generateNormRange(); 
}

__host__ bool FunctorBase::parametersChanged () const {
  if (!cachedParams) return true; 

  std::set<Variable*> params;
  getParameters(params); 
  int counter = 0; 
  for (std::set<Variable*>::iterator v = params.begin(); v != params.end(); ++v) {
    if (cachedParams[counter++] != host_params[(*v)->index]) return true;
  }

  return false;
}

__host__ void FunctorBase::storeParameters () const {
  std::set<Variable*> params;
  getParameters(params); 
  if (!cachedParams) cachedParams = new fptype[params.size()]; 

  int counter = 0; 
  for (std::set<Variable*>::iterator v = params.begin(); v != params.end(); ++v) {
    cachedParams[counter++] = host_params[(*v)->index];
  }
}
