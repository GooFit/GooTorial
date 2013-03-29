#ifndef FUNCTOR_WRITER_HH
#define FUNCTOR_WRITER_HH

class FunctorBase; 

void writeToFile (FunctorBase* pdf, const char* fname);
void readFromFile (FunctorBase* pdf, const char* fname);

#endif
