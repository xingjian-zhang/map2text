.PHONY: clean-tmp

map2text/external/largevis/Linux/LargeVis:
	cd map2text/external/largevis/Linux && g++ LargeVis.cpp main.cpp -o LargeVis -lm -pthread -lgsl -lgslcblas -Ofast -march=native -ffast-math -I/opt/homebrew/include -L/opt/homebrew/lib

clean-tmp:
	rm -rf tmp_trainer tmp map2text/model/tmp