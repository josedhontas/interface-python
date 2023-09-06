#
# makefile for guiimp
#
# ddantas 12/3/2020
#

REPO_NAME       = guiimp
MAIN_NAME       = WinMainTk.py
PYTHON          = python3


dox: *.py
	doxygen $(REPO_NAME).dox
	cd dox/latex; pwd; make; evince refman.pdf&

run:
	$(PYTHON) ./$(MAIN_NAME)

