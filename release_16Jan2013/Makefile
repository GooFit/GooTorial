#------------------------------------------------------------------------------
CXX=nvcc
LD=g++  
OutPutOpt = -o
CXXFLAGS     = -O3 -arch=sm_20 

CUDALOCATION = /usr/local/cuda/5.0.35/
CUDAHEADERS = $(CUDALOCATION)/include/
SRCDIR = $(PWD)/FPOINTER
INCLUDES += -I$(CUDAHEADERS) -I$(SRCDIR) -I$(PWD) -I$(PWD)/rootstuff 
LIBS += -L$(CUDALOCATION)/lib64 -lcudart -L$(PWD)/rootstuff -lRootUtils 

# These are for user-level programs that want access to the ROOT plotting stuff, 
# not just the fitting stuff included in the GooFit-local ripped library. 
ROOT_INCLUDES = -I$(ROOTSYS)/include/
ROOT_LIBS     = -L$(ROOTSYS)/lib/ -lCore -lCint -lRIO -lNet -lHist -lGraf -lGraf3d -lGpad -lTree -lRint -lMatrix -lPhysics -lMathCore -pthread -lThread -lMinuit2 -lMinuit -rdynamic -lFoam 

THRUSTO		= wrkdir/Variable.o wrkdir/PdfBuilder.o wrkdir/ThrustPdfFunctorCUDA.o wrkdir/Faddeeva.o wrkdir/FitControl.o wrkdir/FunctorBase.o wrkdir/DataSet.o wrkdir/BinnedDataSet.o wrkdir/UnbinnedDataSet.o wrkdir/FunctorWriter.o 

.SUFFIXES: 

all:	example4b 

%.o:	%.cu
	$(CXX) $(INCLUDES) $(ROOT_INCLUDES) $(CXXFLAGS) -c -o $@ $<

example%:	example%.o 
		$(LD) $(LDFLAGS) $^ $(THRUSTO) $(LIBS) $(ROOT_LIBS) $(OutPutOpt) $@
		@echo "$@ done"

exercise%:	exercise%.o 
		$(LD) $(LDFLAGS) $^ $(THRUSTO) $(LIBS) $(ROOT_LIBS) $(OutPutOpt) $@
		@echo "$@ done"

clean:
		@rm -f *.o core 
		cd rootstuff; $(MAKE) clean 
