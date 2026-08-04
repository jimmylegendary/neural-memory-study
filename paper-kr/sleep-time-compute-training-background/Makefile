ROOT := $(abspath ../..)
BUILDER := $(ROOT)/paper-kr/common/build_publications.py

.PHONY: background clean

background:
	python3 $(BUILDER) --root $(ROOT) --target background

clean:
	python3 $(BUILDER) --root $(ROOT) --target background --clean

