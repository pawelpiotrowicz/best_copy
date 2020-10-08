CXX=g++ -Wall 
FL=-march=native 
#FL=-march=native -fsanitize=thread
#FL=-march=native -fsanitize=memory


main: mem_throughput.cc 
	$(CXX) -O3 -fopenmp $(FL) -o prog mem_throughput.cc  -lpthread 

debug: mem_throughput.cc 
	$(CXX) -g -fopenmp $(FL) -o prog mem_throughput.cc  -lpthread

