// The Computer Language Benchmarks Game
// http://benchmarksgame.alioth.debian.org/
//
// contributed by Elam Kolenovic
//
// Changes (2013-05-07)
//   - changed omp schedule for more even distribution of work
//   - changed loop variables to signed mse::msev_size_teger because msvc was complaining
//     when omp was enabled
//   - replaced std::copy and std::fill by one loop. slightly faster.
//   - swapped order of tests in for-i-loop. slightly faster.
//
// Changes (2013-04-19)
//   - using omp
//   - use buffer and fwrite at end instead of putchar
//   - pre-calculate cr0[]
//   - rename variables and use underscore before the index part of the name
//   - inverted bit tests, better performance under MSVC
//   - optional argument for file output, usefull in windows shell
//
// Changes (2013-04-07):
//   - removed unnecessary arrays, faster especially on 32 bits
//   - using putchar instead of iostreams, slightly faster
//   - using namespace std for readability
//   - replaced size_t with unsigned
//   - removed some includes

#include <cstdio>
#include <cstdlib>
#include <vector>

#include <algorithm>
#include <string>
#include <fstream>
#include <iostream>
#define MSE_MSEARRAY_USE_MSE_PRIMITIVES
#define MSE_MSEVECTOR_USE_MSE_PRIMITIVES
#include "msemstdvector.h"
#include "msemstdarray.h"
#include "mseprimitives.h"
#include "mseasyncshared.h"

typedef unsigned char Byte;

using namespace std;

template<class TCR0Pointer>
mse::nii_vector<Byte> foo1(int sy, TCR0Pointer cr0_shimptr, const mse::msev_size_t max_x, const mse::msev_size_t height, const mse::msev_size_t max_iterations, const double limit_sq)
{
	/* First we're going to obtain a (fast) scope pointer to the (shared) cr0 vector. */
	auto xscope_cr0_ptr_store = mse::make_xscope_strong_pointer_store(cr0_shimptr);
	auto cr0_xscpptr = xscope_cr0_ptr_store.xscope_ptr();

	mse::msev_size_t y = sy;
	//Byte* line = &l_buffer[y * max_x];
	//auto line = l_buffer.begin();
	mse::TXScopeObj<mse::nii_vector<Byte> > l_buffer;
	l_buffer.resize(max_x);

	auto line = mse::make_xscope_iterator(&l_buffer);
	//line += y * max_x;

	const double ci0 = 2.0 * double(mse::as_a_size_t(y)) / double(mse::as_a_size_t(height)) - 1.0;

	for (mse::msev_size_t x = 0; x < max_x; ++x)
	{
		//const double* cr0_x = &cr0[8 * x];
		//double cr[8];
		//double ci[8];

		//auto cr0_x = cr0.cbegin();
		//cr0_x += 8 * x;
		mse::nii_array<double, 8> cr0_x{ (*cr0_xscpptr)[8 * x], (*cr0_xscpptr)[8 * x + 1], (*cr0_xscpptr)[8 * x + 2], (*cr0_xscpptr)[8 * x + 3], (*cr0_xscpptr)[8 * x + 4], (*cr0_xscpptr)[8 * x + 5], (*cr0_xscpptr)[8 * x + 6], (*cr0_xscpptr)[8 * x + 7] };
		mse::nii_array<double, 8> cr;
		mse::nii_array<double, 8> ci;

		for (mse::msear_size_t k = 0; k < 8; ++k)
		{
			cr[k] = cr0_x[k];
			ci[k] = ci0;
		}

		Byte bits = 0xFF;
		for (mse::msear_size_t i = 0; bits && i < max_iterations; ++i)
		{
			Byte bit_k = 0x80;
			for (mse::msear_size_t k = 0; k < 8; ++k)
			{
				if (bits & bit_k)
				{
					const double cr_k = cr[k];
					const double ci_k = ci[k];
					const double cr_k_sq = cr_k * cr_k;
					const double ci_k_sq = ci_k * ci_k;

					cr[k] = cr_k_sq - ci_k_sq + cr0_x[k];
					ci[k] = 2.0 * cr_k * ci_k + ci0;

					if (cr_k_sq + ci_k_sq > limit_sq)
					{
						bits ^= bit_k;
					}
				}
				bit_k >>= 1;
			}
		}
		line[x] = bits;
	}
	return l_buffer;
}

int main(int argc, char* argv[])
{
	const mse::msev_size_t    N = std::max(0, (argc > 1) ? stoi(argv[1]) : 0);
	const mse::msev_size_t    width = N;
	const mse::msev_size_t    height = N;
	const mse::msev_size_t    max_x = (width + 7) / 8;
	const mse::msev_size_t    max_iterations = 50;
	const double limit = 2.0;
	const double limit_sq = limit * limit;

	const Byte zbyte = 0;
	//mse::mstd::vector<Byte> buffer(height * max_x, zbyte);
	mse::TXScopeObj<mse::nii_vector<Byte> > buffer(height * max_x, zbyte);

	const double zdouble = 0.0;
	//mse::mstd::vector<double> cr0(8 * max_x, zdouble);
	mse::TXScopeObj<mse::nii_vector<double> > cr0(8 * max_x, zdouble);

	/* We won't bother (safely) parallelizing this loop for now. It probably wouldn't have much effect anyway. */
	for (mse::msev_size_t x = 0; x < max_x; ++x)
	{
		for (mse::msev_size_t k = 0; k < 8; ++k)
		{
			const mse::msev_size_t xk = 8 * x + k;
			cr0[xk] = (2.0 * double(mse::as_a_size_t(xk))) / double(mse::as_a_size_t(width)) - 1.5;
		}
	}

	/* making a copy of cr0 that can be safely accessed from multiple threads */
	auto cr0_shimptr = mse::make_asyncsharedv2immutable<mse::nii_vector<double>>(cr0);

	const auto sheight = height;
	mse::mstd::vector<std::future<mse::nii_vector<Byte>>> futures;
	for (int sy = 0; sy < sheight; ++sy) {
		futures.emplace_back(mse::mstd::async(foo1<decltype(cr0_shimptr)>, sy, cr0_shimptr, max_x, height, max_iterations, limit_sq));
	}
	auto target_xscpiter = mse::make_xscope_iterator(&buffer);
	for (auto it = futures.begin(); futures.end() != it; it++) {
		mse::TXScopeObj<mse::nii_vector<Byte> > src_line = (*it).get();

		auto src_begin_xscpiter = mse::make_xscope_iterator(&src_line);
		auto src_end_xscpiter = src_begin_xscpiter + mse::as_a_size_t(src_line.size());
		std::move(src_begin_xscpiter, src_end_xscpiter, target_xscpiter);

		target_xscpiter += max_x;

		//checksumOutput[depthIteration] = (*it).get();
	}

	{
		ofstream file1;
		if (3 <= argc) {
			file1.open(argv[2], ios::out | ios::binary);
		}
		ostream& out = file1.good() ? file1 : std::cout;

		out << "P4\n" << mse::as_a_size_t(width) << " " << mse::as_a_size_t(height) << "\n";
		auto xscope_buffer_proxy = mse::make_xscope_random_access_section(&buffer);
		for (const auto byte : xscope_buffer_proxy) {
			out.put(byte);
		}
	}

	return 0;
}
