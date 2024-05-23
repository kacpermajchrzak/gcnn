# DIR_NAME = .
# FILE_NAME = logic.md

OUTPUT_NAME = "./sp2-project-report-majchrzak-mazur.pdf"

# pdf:
# 	jupyter nbconvert \
# 		--to markdown logic.ipynb \
# 		-o logic.md \
# 		--TagRemovePreprocessor.remove_cell_tags='{"hide_code"}'
# 	pandoc $(DIR_NAME)/$(FILE_NAME) \
# 		-o $(OUTPUT_NAME) \
# 		--pdf-engine=xelatex \
# 		--metadata-file=conf.yaml \
# 		--metadata date="`date +%d.%m.%Y`" \
# 		--toc \
# 		--toc-depth=1 \
# 		-N
		
# 	@ echo ""
# 	@ echo 🌟🌟🌟 Result: $(OUTPUT_NAME) generated 🌟🌟🌟

TITLE = "Śledzenie obiektów z wykorzystaniem granulacji i głębokich sieci neuronowych" 
SUBTITLE = "Projekt na przedmiot Studio Projektowe 2 na podstawie publikacji 'Granulated deep learning and Z-numbers in motion detection and object recognition'"
AUTHOR = "Kacper Majchrzak, Mateusz Mazur"
DATE = `date "+%d.%m.%Y"`


REPORT_NAME = "report" #Projekt_GRA_OCENA_IMIE_NAZWISKO

pdf:
	jupyter nbconvert \
		--to pdf logic.ipynb \
		--output $(OUTPUT_NAME) \
		--LatexPreprocessor.title $(TITLE) \
		--LatexPreprocessor.subtitle $(SUBTITLE) \
		--LatexPreprocessor.date $(DATE) \
		--LatexPreprocessor.author_names $(AUTHOR) \
		--TagRemovePreprocessor.remove_cell_tags='{"hide_cell"}'
		@echo ""
