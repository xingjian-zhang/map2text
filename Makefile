llm4explore/external/largevis/Linux/LargeVis:
	cd llm4explore/external/largevis/Linux && g++ LargeVis.cpp main.cpp -o LargeVis -lm -pthread -lgsl -lgslcblas -Ofast -march=native -ffast-math -I/opt/homebrew/include -L/opt/homebrew/lib
