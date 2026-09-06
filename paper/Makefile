TECTONIC ?= $(HOME)/.local/bin/tectonic
CLEANER  ?= ../.venv/bin/arxiv_latex_cleaner

.PHONY: pdf data arxiv check clean

pdf: main.pdf
main.pdf: main.tex refs.bib data/*.csv
	$(TECTONIC) -X compile main.tex

data:                        # regenerate figure CSVs from saved artifacts
	cd .. && .venv/bin/python figs.py

arxiv: pdf                   # strip comments/unused files -> ../paper_arXiv/, verify it compiles, ship .bbl
	rm -rf ../paper_arXiv
	cd .. && $(subst ../,,$(CLEANER)) paper --keep_bib
	cd ../paper_arXiv && rm -f Makefile && $(TECTONIC) -X compile --keep-intermediates main.tex \
	  && rm -f main.pdf main.aux main.log main.blg main.xdv && ls
	cd .. && tar czf paper_arXiv.tar.gz paper_arXiv && ls -la paper_arXiv.tar.gz


# --- writing QA (tools: ~/.local/bin/{chktex,tex-fmt}; skills: ~/.local/share/academic-writing-skills)
CHKTEX  ?= $(HOME)/.local/bin/chktex -l $(HOME)/.local/etc/chktexrc -q -I0 -n 1 -n 2 -n 3 -n 8 -n 12 -n 13 -n 24 -n 25 -n 31 -n 36 -n 44
TEXFMT  ?= $(HOME)/.local/bin/tex-fmt --wraplen 100
SKILLS  ?= $(HOME)/.local/share/academic-writing-skills/academic-writing-skills

.PHONY: lint fmt audit
lint:                        # chktex (pedantic spacing rules and template noise suppressed)
	$(CHKTEX) main.tex

fmt:                         # tex-fmt in place
	$(TEXFMT) main.tex

audit:                       # latex-paper-en + paper-audit scripts -> review/
	mkdir -p review
	uv run -q python -B $(SKILLS)/latex-paper-en/scripts/verify_bib.py refs.bib --tex main.tex > review/verify_bib.txt
	uv run -q python -B $(SKILLS)/latex-paper-en/scripts/analyze_sentences.py main.tex > review/analyze_sentences.txt
	uv run -q python -B $(SKILLS)/latex-paper-en/scripts/analyze_logic.py main.tex --motivation-thread > review/analyze_logic.txt
	uv run -q python -B $(SKILLS)/latex-paper-en/scripts/analyze_experiment.py main.tex > review/analyze_experiment.txt
	uv run -q python -B $(SKILLS)/latex-paper-en/scripts/deai_check.py main.tex --tier medium --analyze > review/deai_check.txt
	uv run -q python -B $(SKILLS)/paper-audit/scripts/audit.py main.tex --mode gate --venue iclr --format md > review/gate.md || true
	@sed -n '/Verdict/,/Advisory/p' review/gate.md
check:                       # page count and overfull boxes
	@pdfinfo main.pdf | grep Pages
	@$(TECTONIC) -X compile main.tex 2>&1 | grep -ci overfull || true

clean:
	rm -f main.pdf *.aux *.log *.bbl *.blg *.png
