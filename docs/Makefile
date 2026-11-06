SPHINXBUILD   = sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: help html clean

help:
	@$(SPHINXBUILD) -M help $(SOURCEDIR) $(BUILDDIR) $(SPHINXOPTS) $(O)

html:
	if [ -d ../examples ]; then test -L examples || ln -sf ../examples examples; fi
	$(SPHINXBUILD) -b html $(SOURCEDIR) $(BUILDDIR)/html $(SPHINXOPTS) $(O)
	@echo
	@echo "Build finished. Open $(BUILDDIR)/html/index.html to view."

clean:
	rm -rf $(BUILDDIR) generated examples
