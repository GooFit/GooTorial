#include "FunctorWriter.hh"
#include <fstream> 
#include <map>
#include "FunctorBase.hh" 

void writeToFile (FunctorBase* pdf, const char* fname) {
  FunctorBase::parCont params;
  pdf->getParameters(params); 

  std::ofstream writer;
  writer.open(fname);

  for (FunctorBase::parIter p = params.begin(); p != params.end(); ++p) {
    writer << (*p)->name << " " 
	   << (*p)->value << " " 
	   << (*p)->error << " "
	   << (*p)->numbins << " "
	   << (*p)->lowerlimit << " "
	   << (*p)->upperlimit 
	   << std::endl; 
  }

  writer.close(); 
}


void readFromFile (FunctorBase* pdf, const char* fname) {
  FunctorBase::parCont params;
  pdf->getParameters(params); 

  std::map<string, Variable*> tempMap;
  for (FunctorBase::parIter p = params.begin(); p != params.end(); ++p) {
    tempMap[(*p)->name] = (*p); 
  }  

  std::ifstream reader;
  reader.open(fname);
  std::string buffer;
  char discard[1000]; 
  int numSet = 0; 
  while (true) {
    reader >> buffer;
    if (reader.eof()) break; 
    Variable* var = tempMap[buffer];
    if (var) {
      reader >> var->value
	     >> var->error
	     >> var->numbins 
	     >> var->lowerlimit
	     >> var->upperlimit; 
      if (++numSet == tempMap.size()) break; 
    }
    else {
      reader.getline(discard, 1000); 
    }
  }

  reader.close(); 
}
