#------------------------------------------------------------------------------
CXX=nvcc
LD=g++  
OutPutOpt = -o
CXXFLAGS     = -O3 -arch=sm_20 

CUDALIBDIR=lib64
UNAME=$(shell uname)
ifeq ($(UNAME), Darwin)
CXXFLAGS+=-m64
CUDALIBDIR=lib
endif

GOODIR = $(PWD)/release_16Jan2013
EXLIST = example2 example3a example3b example3c example4a example4b example4c example4d example4e

.SUFFIXES: 

examples:	$(EXLIST)

include $(GOODIR)/Makefile.goofit 

%.o:	%.cu
	$(CXX) $(INCLUDES) $(ROOT_INCLUDES) $(CXXFLAGS) -c -o $@ $<

example%:	example%.o $(THRUSTO) $(ROOTUTILLIB) 
		$(LD) $(LDFLAGS) $< $(THRUSTO) $(LIBS) $(ROOT_LIBS) $(OutPutOpt) $@
		@echo "$@ done"

exercise%:	exercise%.o $(THRUSTO) $(ROOTUTILLIB) 
		$(LD) $(LDFLAGS) $< $(THRUSTO) $(LIBS) $(ROOT_LIBS) $(OutPutOpt) $@
		@echo "$@ done"

exclean:
		@rm -f *.o $(EXLIST) 
