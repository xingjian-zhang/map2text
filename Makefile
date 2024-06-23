.PHONY: clean-tmp

llm4explore/external/largevis/Linux/LargeVis:
	cd llm4explore/external/largevis/Linux && g++ LargeVis.cpp main.cpp -o LargeVis -lm -pthread -lgsl -lgslcblas -Ofast -march=native -ffast-math -I/opt/homebrew/include -L/opt/homebrew/lib

clean-tmp:
	rm -rf tmp_trainer tmp llm4explore/model/tmp