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
#include "msemsevector.h"
#include "msemsearray.h"

typedef unsigned char Byte;

using namespace std;

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
	mse::msevector<Byte> buffer(height * max_x, zbyte);

	const double zdouble = 0.0;
	mse::msevector<double> cr0(8 * max_x, zdouble);

	/* omp requires loop variables be signed ints. */
	const int smax_x = max_x;
#pragma omp parallel for
	for (int sx = 0; sx < smax_x; ++sx)
	{
		mse::msev_size_t x = sx;
		for (mse::msev_size_t k = 0; k < 8; ++k)
		{
			const mse::msev_size_t xk = 8 * x + k;
			cr0[xk] = (2.0 * xk) / width - 1.5;
		}
	}

	const int sheight = height;
#pragma omp parallel for schedule(guided)
	for (int sy = 0; sy < sheight; ++sy)
	{
		mse::msev_size_t y = sy;
		//Byte* line = &buffer[y * max_x];
		auto line = buffer.ss_begin();
		line += y * max_x;

		const double ci0 = 2.0 * y / height - 1.0;

		for (mse::msev_size_t x = 0; x < max_x; ++x)
		{
			//const double* cr0_x = &cr0[8 * x];
			//double cr[8];
			//double ci[8];

			//auto cr0_x = cr0.ss_cbegin();
			//cr0_x += 8 * x;
			mse::msearray<double, 8> cr0_x{ cr0[8 * x], cr0[8 * x + 1], cr0[8 * x + 2], cr0[8 * x + 3], cr0[8 * x + 4], cr0[8 * x + 5], cr0[8 * x + 6], cr0[8 * x + 7] };
			mse::msearray<double, 8> cr;
			mse::msearray<double, 8> ci;

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
						auto& cr_k_ref = cr[k];
						auto& ci_k_ref = ci[k];
						const double cr_k = cr_k_ref;
						const double ci_k = ci_k_ref;
						const double cr_k_sq = cr_k * cr_k;
						const double ci_k_sq = ci_k * ci_k;

						cr_k_ref = cr_k_sq - ci_k_sq + cr0_x[k];
						ci_k_ref = 2.0 * cr_k * ci_k + ci0;

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
	}

	{
		ofstream file1;
		if (3 <= argc) {
			file1.open(argv[2], ios::out | ios::binary);
		}
		ostream& out = file1.good() ? file1 : std::cout;

		out << "P4\n" << width << " " << height << "\n";
		for (const auto& byte : buffer) {
			out.put(byte);
		}
	}

	return 0;
}
