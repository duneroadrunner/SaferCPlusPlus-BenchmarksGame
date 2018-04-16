// The Computer Language Benchmarks Game
// http://benchmarksgame.alioth.debian.org/
//
// Original C contributed by Sebastien Loisel
// Conversion to C++ by Jon Harrop
// OpenMP parallelize by The Anh Tran
// Add SSE by The Anh Tran
// Additional SSE optimization by Krzysztof Jakubowski

// g++ -pipe -O3 -march=native -fopenmp -mfpmath=sse -msse2 ./spec.c++ -o ./spec.run

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
#define MSE_MSEARRAY_USE_MSE_PRIMITIVES 1
#include "msemsearray.h"
#include "mseasyncshared.h"
#include "mseprimitives.h"
#include "msescope.h"
#include "msemstdvector.h"

#define MAX_N 10000
typedef mse::nii_array<double, MAX_N> double_array_buffer2_type;

template <bool modei> int Index(mse::msear_size_t i, mse::msear_size_t j) {
	return int(((i + j) * (i + j + 1)) >> 1) + int(mse::as_a_size_t(modei ? i : j)) + 1;
}

template <bool modei, class TSrcArrayConstSection, class TDstArraySection>
void EvalPart_V2(TSrcArrayConstSection src_section, TDstArraySection dst_section, mse::msear_size_t dst_section_absolute_start_index) {
	mse::msear_size_t index = 0U;
	/* Even though we're only writing to the "dst_section" span, the "i" variable needs to be relative to the start
	of the whole array, not the span. */
	mse::msear_size_t i = dst_section_absolute_start_index;

	for (; index + 1U < dst_section.size(); index += 2U, i += 2U) {
		__m128d sum = _mm_set_pd(
			src_section[0] / double(Index<modei>(i + 1, 0)),
			src_section[0] / double(Index<modei>(i + 0, 0)));

		__m128d ti = modei ?
			_mm_set_pd(double(i + 1), double(i + 0)) :
			_mm_set_pd(double(i + 2), double(i + 1));
		__m128d last = _mm_set_pd(
			Index<modei>(i + 1, 0),
			Index<modei>(i + 0, 0));

		for (mse::msear_size_t j = 1; j < src_section.size(); j++) {
			//__m128d idx = last + ti + _mm_set1_pd(j);
			__m128d idx = _mm_add_pd(last, _mm_add_pd(ti, _mm_set1_pd(double(mse::msear_as_a_size_t(j)))));
			last = idx;
			//sum = sum + _mm_set1_pd(src_section[j]) / idx;
			sum = _mm_add_pd(sum, _mm_div_pd(_mm_set1_pd(src_section[j]), idx));
		}

		_mm_storeu_pd(&(dst_section[index + 0]), sum);
	}
	for (; index < dst_section.size(); ++index, ++i) {
		double sum = 0;
		for (mse::msear_size_t j = 0; j < src_section.size(); j++)
			sum += src_section[j] / double(Index<modei>(i, j));
		dst_section[index] = sum;
	}
}

template<class TSrcArrayConstSection, class TDstArraySection>
void EvalATimesU_V2(TSrcArrayConstSection src, TDstArraySection dst, mse::msear_size_t dst_absolute_start_index) {
	EvalPart_V2<1>(src, dst, dst_absolute_start_index);
}
template<class TSrcArrayConstSection, class TDstArraySection>
void EvalAtTimesU_V2(TSrcArrayConstSection src, TDstArraySection dst, mse::msear_size_t dst_absolute_start_index) {
	EvalPart_V2<0>(src, dst, dst_absolute_start_index);
}

template<class TSrcAccessRequester, class TDstAccessRequester>
void EvalATimesU_V2_ar(TSrcAccessRequester src_ar, mse::msear_size_t src_size, TDstAccessRequester dst_ar, mse::msear_size_t dst_absolute_start_index) {
	/* While src_ar.readlock_ptr() gives us (data race and bounds enforced) safe access to section of the array
	buffer we need, we can slightly improve performance by explicitly guaranteeing that the array buffer (and our
	safe access to a section of it) will not be deleted before the end of the scope. */
	auto xscope_src_store = mse::make_xscope_strong_pointer_store(src_ar.readlock_ptr());
	auto src_xscpcptr = xscope_src_store.xscope_ptr();
	auto src_begin_xscpciter = mse::make_xscope_const_iterator(src_xscpcptr);
	auto xscope_src_csection = mse::make_xscope_random_access_const_section(src_begin_xscpciter, src_size);
	/* So xscope_src_csection is just a slightly faster version of the array section interface that src_ar.readlock_ptr()
	gives us. */

	EvalPart_V2<1>(xscope_src_csection, dst_ar.writelock_ra_section(), dst_absolute_start_index);
}
template<class TSrcAccessRequester, class TDstAccessRequester>
void EvalAtTimesU_V2_ar(TSrcAccessRequester src_ar, mse::msear_size_t src_size, TDstAccessRequester dst_ar, mse::msear_size_t dst_absolute_start_index) {
	auto xscope_src_store = mse::make_xscope_strong_pointer_store(src_ar.readlock_ptr());
	auto src_xscpcptr = xscope_src_store.xscope_ptr();
	auto src_begin_xscpciter = mse::make_xscope_const_iterator(src_xscpcptr);
	auto xscope_src_csection = mse::make_xscope_random_access_const_section(src_begin_xscpciter, src_size);

	EvalPart_V2<0>(xscope_src_csection, dst_ar.writelock_ra_section(), dst_absolute_start_index);
}

template<class TBufferAccessRequester, class TSectionSizeList>
void EvalAtTimesU_V2(TBufferAccessRequester src_ar, TBufferAccessRequester dst_ar, TBufferAccessRequester tmp_ar
	, TSectionSizeList section_sizes, mse::msear_size_t N, mse::msear_size_t max_chunk_size, mse::msear_size_t num_threads) {
	{
		/* TXScopeAsyncRASectionSplitter<> will generate a new access requester for each section. */
		mse::TXScopeAsyncRASectionSplitter<decltype(tmp_ar)> ra_section_split1(tmp_ar, section_sizes);

		auto& EvalATimesU_V2_ar_ref = EvalATimesU_V2_ar<decltype(src_ar), decltype(ra_section_split1.ra_section_access_requester(0))>;

		mse::xscope_thread_carrier xscope_threads;
		for (size_t i = 0; i < num_threads; i += 1) {
			auto ar = ra_section_split1.ra_section_access_requester(i);
			xscope_threads.new_thread(EvalATimesU_V2_ar_ref, src_ar, N, ar, i*max_chunk_size);
		}
		/* xscope_thread_carrier ensures that the scope will not end until all the threads have terminated. */
	}
	{
		mse::TXScopeAsyncRASectionSplitter<decltype(dst_ar)> ra_section_split1(dst_ar, section_sizes);
		auto& EvalAtTimesU_V2_ar_ref = EvalAtTimesU_V2_ar<decltype(tmp_ar), decltype(ra_section_split1.ra_section_access_requester(0))>;

		mse::xscope_thread_carrier xscope_threads;
		for (size_t i = 0; i < num_threads; i += 1) {
			auto ar = ra_section_split1.ra_section_access_requester(i);
			xscope_threads.new_thread(EvalAtTimesU_V2_ar_ref, tmp_ar, N, ar, i*max_chunk_size);
		}
	}
}

struct sums1_return_type {
	double sumvb = 0.0;
	double sumvv = 0.0;
};
/* Here "shareable" just means that the type is appropriate for sharing between threads (i.e. basically doesn't contain
any pointers or references). */
typedef mse::us::TUserDeclaredAsyncShareableObj<sums1_return_type> shareable_sums1_return_type;

template<class TBufferAccessRequester, class TSectionSizeList>
shareable_sums1_return_type sums1(TBufferAccessRequester u_access_requester, TBufferAccessRequester v_access_requester
	, mse::msear_size_t start_index, mse::msear_size_t section_size) {
	double sumvb = 0.0, sumvv = 0.0;
	{
		auto xscope_u_store = mse::make_xscope_strong_pointer_store(u_access_requester.readlock_ptr());
		auto u_xscpptr = xscope_u_store.xscope_ptr();
		auto xscope_v_store = mse::make_xscope_strong_pointer_store(v_access_requester.readlock_ptr());
		auto v_xscpptr = xscope_v_store.xscope_ptr();

		for (mse::msear_size_t i = start_index; i < start_index + section_size; i++) {
			sumvv += (*v_xscpptr)[i] * (*v_xscpptr)[i];
			sumvb += (*u_xscpptr)[i] * (*v_xscpptr)[i];
		}
	}
	shareable_sums1_return_type retval;
	retval.sumvb = sumvb;
	retval.sumvv = sumvv;
	return retval;
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
	for (int i = 0; i < CPU_SETSIZE; ++i)
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

	/* Arguably, the "cleanest" way of sharing objects among threads with SaferCPlusPlus involves allocating the object
	on the heap. Pre-C++17, "over-aligned" memory allocation on the heap would require a (non-portable) custom allocator.
	A bit of a hassle. So here we'll allocate the aligned arrays on the stack. And so we'll demonstrate the (not yet
	documented at the time of this writing) use of the library to safely share stack allocated objects. */

	/* First we'll declare the aligned array buffers as "access controlled" scope objects. */
	ALIGNAS(16) mse::TXScopeObj<mse::TXScopeAccessControlledObj<double_array_buffer2_type> > xscope_u2;
	ALIGNAS(16) mse::TXScopeObj<mse::TXScopeAccessControlledObj<double_array_buffer2_type> > xscope_v2;
	ALIGNAS(16) mse::TXScopeObj<mse::TXScopeAccessControlledObj<double_array_buffer2_type> > xscope_tmp2;

	/* So that we can obtain "async access requesters" which will provide (data race) safe access to the array buffers. */
	auto xscope_u2_access_requester = mse::make_xscope_asyncsharedv2acoreadwrite(&xscope_u2);
	auto xscope_v2_access_requester = mse::make_xscope_asyncsharedv2acoreadwrite(&xscope_v2);
	auto xscope_tmp2_access_requester = mse::make_xscope_asyncsharedv2acoreadwrite(&xscope_tmp2);

	double vBv2 = 0.0;
	double vv2 = 0.0;

	{
		auto xscope_u2_store = mse::make_xscope_strong_pointer_store(xscope_u2_access_requester.writelock_ptr());
		auto u2_xscpptr = xscope_u2_store.xscope_ptr();

		mse::msear_size_t begin = 0;
		mse::msear_size_t end = N;

		/* We won't bother to parallelize this part for now. */
		for (mse::msear_size_t i = begin; i < end; i++) {
			(*u2_xscpptr)[i] = 1.0;
		}
	}
	{
		/* First we create a list of a the sizes of each section. We'll use a vector here, but any iteratable container will work. */
		mse::mstd::vector<mse::msear_size_t> section_sizes;
		mse::mstd::vector<mse::msear_size_t> section_start_indexes;

		const mse::CSize_t num_threads = GetThreadCount();
		mse::msear_size_t max_chunk_size = N / num_threads;
		for (mse::CSize_t i = 0U; i < num_threads; i += 1) {
			// calculate each thread's working range [r1 .. r2) => static schedule
			const mse::msear_size_t begin = i * max_chunk_size;
			const mse::msear_size_t end = (i < (num_threads - 1)) ? (begin + max_chunk_size) : N;
			const auto chunk_size = end - begin;

			section_sizes.emplace_back(chunk_size);
			section_start_indexes.emplace_back(begin);
		}

		for (mse::msear_size_t ite = 0; ite < 10; ++ite) {
			EvalAtTimesU_V2(xscope_u2_access_requester, xscope_v2_access_requester, xscope_tmp2_access_requester, section_sizes, N, max_chunk_size, num_threads);
			EvalAtTimesU_V2(xscope_v2_access_requester, xscope_u2_access_requester, xscope_tmp2_access_requester, section_sizes, N, max_chunk_size, num_threads);
		}

		{
			typedef mse::xscope_future_carrier<shareable_sums1_return_type> xscope_futures_t;
			xscope_futures_t xscope_futures;
			mse::mstd::vector<xscope_futures_t::handle_t> future_handles;
			for (size_t i = 0; i < num_threads; i += 1) {
				/* Adding a new future automatically launches an associated (std::async()) task with the given parameters. */
				auto handle = xscope_futures.new_future(sums1<decltype(xscope_u2_access_requester), decltype(xscope_v2_access_requester)>
					, xscope_u2_access_requester, xscope_v2_access_requester, section_start_indexes[i], section_sizes[i]);
				future_handles.push_back(handle);
			}
			for (auto it = future_handles.begin(); future_handles.end() != it; it++) {
				auto res = xscope_futures.xscope_ptr_at(*it)->get();
				vBv2 += res.sumvb;
				vv2 += res.sumvv;
			}
		}
	}

	return sqrt(vBv2 / vv2);
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

