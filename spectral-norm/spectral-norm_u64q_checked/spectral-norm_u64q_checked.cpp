// The Computer Language Benchmarks Game
// http://benchmarksgame.alioth.debian.org/
//
// Original C contributed by Sebastien Loisel
// Conversion to C++ by Jon Harrop
// OpenMP parallelize by The Anh Tran
// Add SSE by The Anh Tran
// Additional SSE optimization by Krzysztof Jakubowski

// g++ -pipe -O3 -march=native -fopenmp -mfpmath=sse -msse2 \
//     ./spec.c++ -o ./spec.run

#ifdef _MSC_VER
#include "windows.h"
#undef max
#undef min
#define RESTRICT __restrict
#define ALIGNAS(x) __declspec(align(x))
#else // _MSC_VER
#include <sched.h>
#define RESTRICT __restrict__
#define ALIGNAS(x) __attribute__((aligned(x)))
//define ALIGNAS(x) alignas(x)
#endif // _MSC_VER

#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <omp.h>
#include <emmintrin.h>

#include <string>
#include <iomanip>
#include <iostream>
#include <vector>
#include "msemsearray.h"
#include "mseasyncshared.h"

/* This "safer" implementation contains some considerable modifications from the original. Most of the
modifications are to add automatic access controls to elements that are being accessed from asynchronous
threads. Data race prevention doesn't seem to be a priority of OMP. Excluding the ones for multithreading
safety, there are actually relatively few code changes from the original.  */

#define MAX_N 10000
typedef ALIGNAS(16) mse::msearray<double, MAX_N> al_double_array_buffer_type;

/* The original implementation partitioned the arrays of doubles into chunks, each assigned to a thread.
Even though each chunk is only accessed by a single thread for writing, each chunk is accessed by all
threads for reading. Especially since these reads and writes are interleaved, prudent programming practices
would have us protect these chunks with automatic access controls. To make this cleaner we define an "array
span" data type that provides access to a specified chunk of an array. */
template <class _TElement, class _TArray, int _Size>
class quickndirty_index_span_type {
public:
	quickndirty_index_span_type(_TArray& array, mse::msear_size_t start_index, mse::msear_size_t size) : m_array_ref(array), m_start_index(start_index), m_size(size) {
		if (_Size < (start_index + size)) { std::terminate(); }
	}
	quickndirty_index_span_type(quickndirty_index_span_type& src_span, mse::msear_size_t start_index, mse::msear_size_t size) : m_array_ref(src_span.m_array_ref), m_start_index(start_index), m_size(size) {
		if (src_span.size() < (start_index + size)) { std::terminate(); }
	}
	_TElement& operator[](mse::msear_size_t index) {
		if (m_size <= index) { std::terminate(); }
		return m_array_ref[m_start_index + index];
	}
	const _TElement& operator[](mse::msear_size_t index) const {
		if (m_size <= index) { std::terminate(); }
		return m_array_ref[m_start_index + index];
	}
	mse::msear_size_t size() const { return m_size; }
	const mse::msear_size_t m_start_index = 0;
	const mse::msear_size_t m_size = 0;
	_TArray& m_array_ref;
};

typedef quickndirty_index_span_type<double, al_double_array_buffer_type, MAX_N> sn_span_type;
typedef sn_span_type sn_array_accessor_type;

template <bool modei> int Index(mse::msear_size_t i, mse::msear_size_t j) {
	return (((i + j) * (i + j + 1)) >> 1) + (modei ? i : j) + 1;
}

template <bool modei>
void EvalPart(const sn_array_accessor_type& RESTRICT src, sn_span_type& RESTRICT dst) {
	mse::msear_size_t index = 0U;
	/* Even though we're only writing to the "dst" span, the "i" variable needs to be relative to the start
	of the whole array, not the span. */
	mse::msear_size_t i = dst.m_start_index;

	for (; index + 1U < dst.size(); index += 2U, i += 2U) {
		__m128d sum = _mm_set_pd(
			src[0] / double(Index<modei>(i + 1, 0)),
			src[0] / double(Index<modei>(i + 0, 0)));

		__m128d ti = modei ?
			_mm_set_pd(i + 1, i + 0) :
			_mm_set_pd(i + 2, i + 1);
		__m128d last = _mm_set_pd(
			Index<modei>(i + 1, 0),
			Index<modei>(i + 0, 0));

		for (mse::msear_size_t j = 1; j < src.size(); j++) {
			//__m128d idx = last + ti + _mm_set1_pd(j);
			__m128d idx = _mm_add_pd(last, _mm_add_pd(ti, _mm_set1_pd(j)));
			last = idx;
			//sum = sum + _mm_set1_pd(src[j]) / idx;
			sum = _mm_add_pd(sum, _mm_div_pd(_mm_set1_pd(src[j]), idx));
		}

		_mm_storeu_pd(&(dst[index + 0]), sum);
	}
	for (; index < dst.size(); ++index, ++i) {
		double sum = 0;
		for (mse::msear_size_t j = 0; j < src.size(); j++)
			sum += src[j] / double(Index<modei>(i, j));
		dst[index] = sum;
	}

}

void EvalATimesU(const sn_array_accessor_type& src, sn_span_type& dst) {
	EvalPart<1>(src, dst);
}

void EvalAtTimesU(const sn_array_accessor_type& src, sn_span_type& dst) {
	EvalPart<0>(src, dst);
}

typedef std::vector<mse::TAsyncSharedObjectThatYouAreSureHasNoUnprotectedMutablesReadWriteConstPointer<sn_span_type>> readlock_ptrs_type;
readlock_ptrs_type obtain_readlocks_on_all_subspans(std::vector<mse::TAsyncSharedObjectThatYouAreSureHasNoUnprotectedMutablesReadWriteAccessRequester<sn_span_type>>& access_requesters) {
	readlock_ptrs_type retval;
	for (auto& access_requester : access_requesters) {
		retval.emplace_back(access_requester.readlock_ptr());
	}
	return retval;
}

void EvalAtATimesU(const sn_array_accessor_type& src, const sn_array_accessor_type& dst, const sn_array_accessor_type& tmp,
	std::vector<mse::TAsyncSharedObjectThatYouAreSureHasNoUnprotectedMutablesReadWriteAccessRequester<sn_span_type>>& src_subspan_access_requesters,
	std::vector<mse::TAsyncSharedObjectThatYouAreSureHasNoUnprotectedMutablesReadWriteAccessRequester<sn_span_type>>& dst_subspan_access_requesters,
	std::vector<mse::TAsyncSharedObjectThatYouAreSureHasNoUnprotectedMutablesReadWriteAccessRequester<sn_span_type>>& tmp_subspan_access_requesters,
	size_t thread_num) {
		{
			auto tmp_span_writelock_ptr = tmp_subspan_access_requesters.at(thread_num).writelock_ptr();
			auto src_readlock_ptrs = obtain_readlocks_on_all_subspans(src_subspan_access_requesters);
			EvalATimesU(src, *tmp_span_writelock_ptr);
		}
#pragma omp barrier
		{
			auto dst_span_writelock_ptr = dst_subspan_access_requesters.at(thread_num).writelock_ptr();
			auto tmp_readlock_ptrs = obtain_readlocks_on_all_subspans(tmp_subspan_access_requesters);
			EvalAtTimesU(tmp, *dst_span_writelock_ptr);
		}
#pragma omp barrier
		}


int GetThreadCount() {
#ifdef _MSC_VER
	SYSTEM_INFO sysinfo;
	GetSystemInfo(&sysinfo);
	int numCPU = sysinfo.dwNumberOfProcessors;
	return numCPU;
#else // _MSC_VER
	cpu_set_t cs;
	CPU_ZERO(&cs);
	sched_getaffinity(0, sizeof(cs), &cs);

	int count = 0;
	for (mse::msear_size_t i = 0; i < CPU_SETSIZE; ++i)
		if (CPU_ISSET(i, &cs))
			++count;

	return count;
#endif // _MSC_VER
}

struct double_container_type {
	double m_value = 0.0;
};

double spectral_game(mse::msear_size_t N) {
	//ALIGNAS(16) double u[N];
	//ALIGNAS(16) double v[N], tmp[N];
	al_double_array_buffer_type u_buffer;
	al_double_array_buffer_type v_buffer;
	al_double_array_buffer_type tmp_buffer;

	sn_array_accessor_type u(u_buffer, 0, N);
	sn_array_accessor_type v(v_buffer, 0, N);
	sn_array_accessor_type tmp(tmp_buffer, 0, N);

	auto vBv_access_requester = mse::make_asyncsharedreadwrite<double_container_type>();
	auto vv_access_requester = mse::make_asyncsharedreadwrite<double_container_type>();

	const size_t num_threads = GetThreadCount();
	mse::msear_size_t max_chunk_size = N / num_threads;
	std::vector<mse::TAsyncSharedObjectThatYouAreSureHasNoUnprotectedMutablesReadWriteAccessRequester<sn_span_type>> u_subspan_access_requesters;
	std::vector<mse::TAsyncSharedObjectThatYouAreSureHasNoUnprotectedMutablesReadWriteAccessRequester<sn_span_type>> v_subspan_access_requesters;
	std::vector<mse::TAsyncSharedObjectThatYouAreSureHasNoUnprotectedMutablesReadWriteAccessRequester<sn_span_type>> tmp_subspan_access_requesters;
	for (size_t i = 0U; i < num_threads; i += 1) {
		// calculate each thread's working range [r1 .. r2) => static schedule
		const mse::msear_size_t begin = i * max_chunk_size;
		const mse::msear_size_t end = (i < (num_threads - 1)) ? (begin + max_chunk_size) : N;
		const auto chunk_size = end - begin;

		u_subspan_access_requesters.emplace_back(mse::make_asyncsharedobjectthatyouaresurehasnounprotectedmutablesreadwrite<sn_span_type>(u, begin, chunk_size));
		v_subspan_access_requesters.emplace_back(mse::make_asyncsharedobjectthatyouaresurehasnounprotectedmutablesreadwrite<sn_span_type>(v, begin, chunk_size));
		tmp_subspan_access_requesters.emplace_back(mse::make_asyncsharedobjectthatyouaresurehasnounprotectedmutablesreadwrite<sn_span_type>(tmp, begin, chunk_size));
	}

#pragma omp parallel default(shared) num_threads(GetThreadCount())
	{
		// this block will be executed by NUM_THREADS
		// variable declared in this block is private for each thread
		size_t threadid = omp_get_thread_num();

		auto& u_subspan_access_requester = u_subspan_access_requesters.at(threadid);
		auto& v_subspan_access_requester = v_subspan_access_requesters.at(threadid);
		{
			auto u_subspan_writelock_ptr = u_subspan_access_requester.writelock_ptr();
			auto& u_subspan_ref = (*u_subspan_writelock_ptr);
			for (mse::msear_size_t i = 0; i < u_subspan_ref.size(); i++) {
				u_subspan_ref[i] = 1.0;
			}
		}
#pragma omp barrier

		for (mse::msear_size_t ite = 0; ite < 10; ++ite) {
			EvalAtATimesU(u, v, tmp, u_subspan_access_requesters, v_subspan_access_requesters, tmp_subspan_access_requesters, threadid);
			EvalAtATimesU(v, u, tmp, v_subspan_access_requesters, u_subspan_access_requesters, tmp_subspan_access_requesters, threadid);
		}

		double sumvb = 0.0, sumvv = 0.0;
		{
			auto u_span_readlock_ptr = u_subspan_access_requester.readlock_ptr();
			const auto& u_subspan_ref = (*u_span_readlock_ptr);
			auto v_span_readlock_ptr = v_subspan_access_requester.readlock_ptr();
			const auto& v_subspan_ref = (*v_span_readlock_ptr);
			for (mse::msear_size_t i = 0; i < v_subspan_ref.size(); i++) {
				sumvv += v_subspan_ref[i] * v_subspan_ref[i];
				sumvb += u_subspan_ref[i] * v_subspan_ref[i];
			}
		}

#pragma omp critical
		{
			(*(vBv_access_requester.writelock_ptr())).m_value += sumvb;
			(*(vv_access_requester.writelock_ptr())).m_value += sumvv;
		}
	}

	const auto vBv = (*(vBv_access_requester.readlock_ptr())).m_value;
	const auto vv = (*(vv_access_requester.readlock_ptr())).m_value;
	return sqrt(vBv / vv);
}

int main(int argc, char *argv[]) {
	mse::msear_size_t N = ((argc >= 2) ? std::stoi(argv[1]) : 2000);
	if (MAX_N < N) {
		std::cout << "Parameter values larger than " << MAX_N << " are not supported. \n";
		return -1;
	}
	std::cout.precision(9);
	std::cout.setf(std::ios::fixed, std::ios::floatfield); // floatfield set to fixed
	std::cout << spectral_game(N) << '\n';
	return 0;
}

