FreeReg -> IEEE TMI LaTeX source package
==========================================
Title: FreeReg: Free-Coordinate Registration of 3D Models for
       Augmented Reality in Image-Guided Bronchoscopy
Main manuscript: freereg_tbme_full.tex  (10 pages)
Compile: pdflatex freereg_tbme_full ; bibtex freereg_tbme_full ; pdflatex x2
Required: ieeecolor2.cls generic.sty LOGO-generic-web.eps IEEEtran.bst
          references.bib arch_freereg.pdf figs/*.png
Auxiliary (ScholarOne supporting docs): submission/*.tex
  incl. ethics_statement.tex and irb_data_governance.tex
  (retrospective de-identified data under hospital data-governance; no IRB number)
Figures use \graphicspath{{figs/}{./}}.
