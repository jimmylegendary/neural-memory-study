ROOT := $(abspath ../..)
BUILDER := $(ROOT)/paper-kr/common/build_publications.py

.PHONY: full full-draft conference appendix all clean

full:
	python3 $(BUILDER) --root $(ROOT) --target full

full-draft: full

conference:
	python3 $(BUILDER) --root $(ROOT) --target conference

appendix:
	python3 $(BUILDER) --root $(ROOT) --target appendix

all:
	python3 $(BUILDER) --root $(ROOT) --target all

clean:
	python3 $(BUILDER) --root $(ROOT) --target all --clean

