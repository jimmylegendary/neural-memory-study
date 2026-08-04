ROOT := $(abspath ../..)
BUILDER := $(ROOT)/paper-kr/common/build_publications.py

.PHONY: background glossary clean

glossary:
	python3 build_glossary.py --registry $(ROOT)/paper-kr/common/concept-registry.json --output glossary.tex

background: glossary
	python3 $(BUILDER) --root $(ROOT) --target background

clean:
	python3 $(BUILDER) --root $(ROOT) --target background --clean
