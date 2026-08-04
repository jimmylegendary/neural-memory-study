ROOT := $(abspath ../..)
BUILDER := $(ROOT)/paper-kr/common/build_publications.py

.PHONY: assets full full-draft conference appendix all clean

assets:
	python3 figures/build_figures.py
	python3 ../common/build_source_bibliography.py --registry $(ROOT)/research/sleep-time-compute/deep-research/SOURCE-REGISTRY.json --output ../common/source-registry.bib
	python3 build_claim_table.py --claim-map $(ROOT)/claims/stc-study/claim-map.json --output tables/claim-table.tex

full: assets
	python3 $(BUILDER) --root $(ROOT) --target full

full-draft: full

conference: assets
	python3 $(BUILDER) --root $(ROOT) --target conference

appendix: assets
	python3 $(BUILDER) --root $(ROOT) --target appendix

all: assets
	python3 $(BUILDER) --root $(ROOT) --target all

clean:
	python3 $(BUILDER) --root $(ROOT) --target all --clean
