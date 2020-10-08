#include <string.h>
#include <assert.h>
#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include <ctime>
#include <algorithm>
#include <immintrin.h>
#include <functional>
//long mem_size = 4000L*1000*1000;
long mem_size = 4000L*1000*100;
// The number of memory copies in a thread.
// It seems if we have more than one memory copies, the O3 optimization may do something
// to remove some of the copies.
int num_copies = 10;
// The number of threads should be the same as the number of physical cores.
//int num_threads = 16;
int num_threads = 32;

auto start0 = std::chrono::system_clock::now();


void *custom_alloc(size_t size, size_t align=0)
{
	void *ptr;
	
    if(align)
	{
		
		auto ret = posix_memalign(&ptr,align,size);
		if(ret !=0)
		{
			std::cout << "Posix memalign error align=" <<align << " size="<< size << std::endl; 
			exit(1);
		}
		return ptr;
	}

    ptr = malloc(size);
	if(ptr==nullptr)
	{
			std::cout << "malloc error " <<size << std::endl; 
			exit(1);
	} 

	return ptr;
}



void memcpy1_omp(void *dst, void *src, size_t size) {
	assert(((long) src) % sizeof(long) == 0);
	assert(((long) dst) % sizeof(long) == 0);
	assert(size % sizeof(long) == 0);
	long *dst1 = (long *) dst;
	long *src1 = (long *) src;
#pragma omp simd
	for (size_t i = 0; i < size / sizeof(long); i++) {
		dst1[i] = src1[i];
	}
}

void memcpy1(void *dst, void *src, size_t size) {
	assert(((long) src) % sizeof(long) == 0);
	assert(((long) dst) % sizeof(long) == 0);
	assert(size % sizeof(long) == 0);
	long *dst1 = (long *) dst;
	long *src1 = (long *) src;
#pragma simd
	for (size_t i = 0; i < size / sizeof(long); i++) {
		dst1[i] = src1[i];
	}
}




typedef std::function<void(void*,void*,size_t)> MemCpyFn;



void memcpy_avx_next_ver(void *pvDest, void *pvSrc, size_t nBytes) {



    typedef __m256i Reg;

	  if(nBytes<sizeof(Reg))
	  {
		    /* this code can be faster - WIP */
			char* b = reinterpret_cast<char*>(pvSrc);
            char* e = b + nBytes;
			char* out = reinterpret_cast<char*>(pvDest); 
            std::copy(b,e,out); 
	        return; 
	  }	 

    for(size_t i=0; i + sizeof(Reg) <= nBytes;i+=sizeof(Reg)) {
        _mm256_stream_si256(reinterpret_cast<Reg*>(reinterpret_cast<char*>(pvDest)+i), _mm256_stream_load_si256(reinterpret_cast<Reg*>(reinterpret_cast<char*>(pvSrc) + i)));
      } 

    /* We don't care about this scenario at this moment - WIP */  
      size_t left_bytes = nBytes % sizeof(Reg); 

      if(left_bytes)
	  {
		  /* this code can be faster - WIP */ 
	     char* e = reinterpret_cast<char*>(pvSrc) + nBytes;
         char* b = e - left_bytes;
	  	 char* out = reinterpret_cast<char*>(pvDest) + nBytes - left_bytes; 
         std::copy(b,e,out);  				  
	  }

}





void memcpy_mm_stream_si64(void *pvDest, void *pvSrc, size_t nBytes) {

      typedef long long __int64;
      
      /* We don't care about this scenario at this moment - WIP */  
      if(nBytes<sizeof(__int64_t))
	  {
		    /* this code can be faster - WIP */
			char* b = reinterpret_cast<char*>(pvSrc);
            char* e = b + nBytes;
			char* out = reinterpret_cast<char*>(pvDest); 
            std::copy(b,e,out); 
	        return; 
	  }	  

      /* Below is real gain */
      for(size_t i=0; i + sizeof(__int64) <= nBytes;i+=sizeof(__int64)) {
		_mm_stream_si64 ((reinterpret_cast<__int64*>(reinterpret_cast<char*>(pvDest)+i)), *(reinterpret_cast<__int64*>((reinterpret_cast<char*>(pvSrc) + i))));
	  } 

      /* We don't care about this scenario at this moment - WIP */  
      size_t left_bytes = nBytes % sizeof(__int64); 

      if(left_bytes)
	  {
		  /* this code can be faster - WIP */ 
	     char* e = reinterpret_cast<char*>(pvSrc) + nBytes;
         char* b = e - left_bytes;
	  	 char* out = reinterpret_cast<char*>(pvDest) + nBytes - left_bytes; 
         std::copy(b,e,out);  				  
	  }

}


void memcpy_avx_stream_align32(void *pvDest, void *pvSrc, size_t nBytes)
{
		   
      typedef __m256d Reg;

      if(nBytes<sizeof(Reg))
	  {
		    /* this code can be faster - WIP */
			char* b = reinterpret_cast<char*>(pvSrc);
            char* e = b + nBytes;
			char* out = reinterpret_cast<char*>(pvDest); 
            std::copy(b,e,out); 
	        return; 
	  }	 


      for(size_t i=0; i + sizeof(Reg) <= nBytes;i+=sizeof(Reg)) {
		
        _mm256_stream_pd(reinterpret_cast<double*>(reinterpret_cast<char*>(pvDest)+i), _mm256_load_pd(reinterpret_cast<double*>(reinterpret_cast<char*>(pvSrc) + i)));
      } 

       size_t left_bytes = nBytes % sizeof(Reg); 

      if(left_bytes)
	  {
		  /* this code can be faster - WIP */ 
	     char* e = reinterpret_cast<char*>(pvSrc) + nBytes;
         char* b = e - left_bytes;
	  	 char* out = reinterpret_cast<char*>(pvDest) + nBytes - left_bytes; 
         std::copy(b,e,out);  				  
	  }

}


void memcpy_avx512_stream_align64(void *pvDest, void *pvSrc, size_t nBytes)
{

    typedef __m512i Reg;

   if(nBytes<sizeof(Reg))
	  {
		    /* this code can be faster - WIP */
			char* b = reinterpret_cast<char*>(pvSrc);
            char* e = b + nBytes;
			char* out = reinterpret_cast<char*>(pvDest); 
            std::copy(b,e,out); 
	        return; 
	  }	 

for(size_t i=0; i + sizeof(Reg) <= nBytes;i+=sizeof(Reg)) {
	
	_mm512_stream_si512 ( reinterpret_cast<Reg*>(reinterpret_cast<char*>(pvDest)+i) , _mm512_stream_load_si512( reinterpret_cast<void*>(reinterpret_cast<char*>(pvSrc) + i))  );
}

	   size_t left_bytes = nBytes % sizeof(Reg); 

      if(left_bytes)
	  {
		  /* this code can be faster - WIP */ 
	     char* e = reinterpret_cast<char*>(pvSrc) + nBytes;
         char* b = e - left_bytes;
	  	 char* out = reinterpret_cast<char*>(pvDest) + nBytes - left_bytes; 
         std::copy(b,e,out);  				  
	  }
}






void seq_copy_mem(double &throughput, size_t alignment, MemCpyFn mcpy)
{
	//char *src = (char *) malloc(mem_size);
	//char *dst = (char *) malloc(mem_size);
     char *src  = (char*) custom_alloc(mem_size,alignment);
	 char *dst  = (char*) custom_alloc(mem_size,alignment);
	if(!src )
	{
         std::cout << "Fail alloc src" <<  std::endl;
		 return;
	}

	if(!dst )
	{
         std::cout << "Fail alloc dst "  << std::endl;
		 return;
	} 

	memset(src, 0, mem_size);
	memset(dst, 0, mem_size);

	auto copy_start = std::chrono::system_clock::now();
	std::chrono::duration<double> rel_start = copy_start - start0;
	//printf("sequential copy start: %f\n", rel_start.count());
	
	auto start = std::chrono::system_clock::now();
	for (int i = 0; i < num_copies; i++) {
		//memcpy1(dst, src, mem_size);
		mcpy(dst, src, mem_size);
	}
	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end-start;
	free(src);
	free(dst);
	throughput = (mem_size * num_copies) / elapsed_seconds.count();

	std::chrono::duration<double> rel_end = end - start0;
	//printf("sequential copy end: %f\n", rel_end.count());
}

std::vector<long> offsets;
//int stride = 400;
int stride = 512; /* alignment cases required this */

void rand_copy_mem(int thread_id, double &throughput, size_t alignment, MemCpyFn mcpy)
{
	long num_offsets = offsets.size() / num_threads;
	long copy_size = num_offsets * stride;
	const long *offset_start = &offsets[thread_id * num_offsets];
	//char *src = (char *) malloc(mem_size);
	//char *dst = (char *) malloc(copy_size);
	char *src = (char*)custom_alloc(mem_size,alignment);
	char *dst = (char*)custom_alloc(mem_size,alignment);
	
    if(!src )
	{
         std::cout << "Fail alloc src" <<  std::endl;
		 return;
	}

	if(!dst )
	{
         std::cout << "Fail alloc dst "  << std::endl;
		 return;
	} 

	memset(src, 0, mem_size);
	memset(dst, 0, copy_size);

	auto copy_start = std::chrono::system_clock::now();
	std::chrono::duration<double> rel_start = copy_start - start0;
	//printf("random copy start: %.3f\n", rel_start.count());
	
	auto start = std::chrono::system_clock::now();
	for (int j = 0; j < num_copies; j++) {
		for (size_t i = 0; i < num_offsets; i++) {
		//	memcpy1(dst + i * stride, src + offset_start[i] * stride, stride);
		
		   mcpy(dst + i * stride, src + offset_start[i] * stride, stride);
		}
	}
	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end-start;
	free(src);
	free(dst);
	throughput = (copy_size * num_copies) / elapsed_seconds.count();

	std::chrono::duration<double> rel_end = end - start0;
	//printf("random copy end: %f\n", rel_end.count());
}


#define RUN

struct Unit {
   std::string name;
   MemCpyFn fn;
   size_t align_mask;
   double thru;
   Unit(const std::string& _name, MemCpyFn _fn , size_t _align_mask=0) : name(_name), fn(_fn), align_mask(_align_mask), thru(0) {}
};


int main(int argc, char **argv)
{

size_t alignment=0;


if(argc != 2)
{
	std::cout << "./prog [alignment|64|32|0]" << std::endl;
	exit(1);
}

alignment=atoi(argv[1]);
if(alignment)
{
	std::cout << "posix_memalign( alignment="<< alignment<< ") has been used" << std::endl;
} else {
	std::cout << "malloc( ... ) has been used instead of posix_memaling" << std::endl;
}
    std::vector<Unit> test_vector = {
		{"plain libc memcpy()", memcpy},
	/*	{"@Da Version with #pragma simd", memcpy1 }, */
		{"@Da Version with #pragma omp simd", memcpy1_omp },
		
	    { "avx512_stream", memcpy_avx512_stream_align64, 64 },
	    { "_mm_stream_si64", memcpy_mm_stream_si64 },
		{ "avx_version1_32align_req", memcpy_avx_stream_align32, 32 }, 
		{ "avx_version2_32align_req" ,memcpy_avx_next_ver,32},
	 }; 

#ifdef RUN

   for(int i=0;i<1;i++)
   {
	   if(i==1)
	   {
		   std::cout << "====== Sorted ======" << std::endl;
	   }
   for(auto& test : test_vector )
   {
     if( test.align_mask && !(test.align_mask & alignment) )
	 {
		 printf("\n######### \n Test %s skiped due to alignment \n######### \n" , test.name.c_str());
		 continue;
	 }

    printf("--------------\n");
	printf("copy with %d threads\n", num_threads);
	std::vector<std::thread *> threads(num_threads);
	std::vector<double> throughputs(num_threads);

	for (int i = 0; i < num_threads; i++) {
		threads[i] = new std::thread(seq_copy_mem, std::ref(throughputs[i]),alignment,test.fn);
	}
	double throughput = 0;
	for (int i = 0; i < num_threads; i++) {
		threads[i]->join();
		delete threads[i];
		throughput += throughputs[i];
	}
	printf("sequential copy throughput: %f GB/s  => %s \n", throughput / 1024 / 1024 / 1024, test.name.c_str());
  
	// Calculate the total number of strides in the memory.
	size_t num_strides = mem_size / stride;
	// Calculate the location of the strides that we want to copy.
	offsets.resize(num_strides / 2 * num_threads);
	for (size_t i = 0; i < offsets.size(); i++) {
			
		
	  // offsets[i] = (rand() % num_strides) & ~0xFFUL;
		offsets[i] = (rand() % num_strides) ;
		
		if(test.align_mask)
		{
		    offsets[i] &= ~(test.align_mask - 1);

		}
		
		
	}
	
	if(i==1)
	for(size_t i=0;i<offsets.size();i+=(offsets.size() / num_threads))
	{      
	      size_t j=(i + offsets.size() / num_threads) -1;
    	  std::sort(offsets.begin()+i, offsets.begin()+j);
	}
    

	for (int i = 0; i < num_threads; i++) {
		threads[i] = new std::thread(rand_copy_mem, i, std::ref(throughputs[i]),alignment,test.fn);
	}
	throughput = 0;
	for (int i = 0; i < num_threads; i++) {
		threads[i]->join();
		delete threads[i];
		throughput += throughputs[i];
	}
	printf("random copy throughput: %f GB/s => %s\n", throughput / 1024 / 1024 / 1024, test.name.c_str());
    if(i==0)
	  test.thru = throughput / 1024 / 1024 / 1024;
   }
   }


    auto it = std::find_if(test_vector.begin(),test_vector.end(),[](Unit& u ){ return u.name.find("@Da")!=std::string::npos; });
    if(it!=test_vector.end())
	{
  		  auto _max = std::max_element(test_vector.begin(),test_vector.end(),[](Unit& u1, Unit& u2) {  return u1.thru < u2.thru; } );

		  if(_max != it)
		  {
			   std::cout << "+--------------------------------------------------+" << std::endl; 
			   std::cout << " The best copy function is  " << _max->name <<  " gain = " << (int)((1 - (it->thru / _max->thru))*100) << "%" << std::endl;
			   std::cout << "+--------------------------------------------------+" << std::endl; 
		  } 

	}


#else

     typedef  char TT;
	 typedef std::vector<TT> V_t; 
	 V_t v1 = {'a','b', 'c','d','a','b', 'c','d', 'a','b', 'c','d','a','b', 'c','d','j','k','a','a','a','a','a','a'};
	//  V_t v1 = {'b', 'c','d','a','b', 'c','d','a','b', 'c','d','a','b', 'c','d', 'a','b', 'c','d','a','b', 'c','d','j','k'};
	// V_t v1 = {'a','a','a','a','a','a','a','a'};
	

   for(auto& test : test_vector )
   {

     if(test.align_mask)
	   continue;

	 V_t v2(v1.size());
       
	 test.fn(v2.data(),v1.data(), v1.size() * sizeof(TT) ); 	 
	
     if(memcmp(v2.data(),v1.data(),sizeof(TT)*v1.size()) != 0)
	 {
		 std::cout << "Wrong implementation "<< test.name << std::endl;
		 exit(1);
	 } else {
		 std::cout << "[OK] "<< test.name << std::endl;
	 }

   }

#endif


}
