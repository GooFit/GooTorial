

$(WRKDIR)/%.cu:		$(SRCDIR)/%.cu $(SRCDIR)/%.hh 
			@echo "#line 0 \"$<\"" > $(WRKDIR)/%.tmp 
			@cat $(WRKDIR)/%.tmp $< > $@
			@rm $(WRKDIR)/%.tmp
			@echo "Created $@"

$(WRKDIR)/CUDAglob.cu:	$(WRKFUNCTORLIST) $(HEADERLIST) 
			@rm -f $@
			@cat $(WRKFUNCTORLIST) > $@ 
