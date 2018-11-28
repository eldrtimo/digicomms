# Minimal makefile for Sphinx documentation

# You can set these variables from the command line.
SPHINXOPTS    = 
SOURCEDIR     = sphinx
HTMLDIR       = docs
PYMODULE      = digicomms

.PHONY: docs
docs:
	sphinx-apidoc -f -o $(SOURCEDIR) $(PYMODULE)
	sphinx-build -b dirhtml $(SOURCEDIR) $(HTMLDIR)
