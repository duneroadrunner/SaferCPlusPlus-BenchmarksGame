/* The Computer Language Benchmarks Game
http://benchmarksgame.alioth.debian.org/

converted to C++ from D by Rafal Rusin
modified by Vaclav Haisman
modified by The Anh to compile with g++ 4.3.2
modified by Branimir Maksimovic
modified by Kim Walisch
modified by Tavis Bohne
made multithreaded by Jeff Wofford on the model of fasta C gcc #7 and fasta Rust #2

compiles with gcc fasta.cpp -std=c++11 -O2
*/

#include <algorithm>
#include <array>
#include <vector>
#include <thread>
#include <mutex>
#include <iostream>
#include <numeric>

/* performance tuning parameter */
#define MSE_REGISTERED_DEFAULT_CACHE_SIZE 6
//define MSE_REGISTERED_INSTRUMENTATION1
#define MSE_MSEARRAY_USE_MSE_PRIMITIVES
#define MSE_MSEVECTOR_USE_MSE_PRIMITIVES

#include <string>
#include <limits>
#include "msemstdarray.h"
#include "msemstdvector.h"
#include "mseasyncshared.h"
#include "msescope.h"

struct IUB
{
	float p;
	char c;
};

const std::string alu =
{
	"GGCCGGGCGCGGTGGCTCACGCCTGTAATCCCAGCACTTTGG"
	"GAGGCCGAGGCGGGCGGATCACCTGAGGTCAGGAGTTCGAGA"
	"CCAGCCTGGCCAACATGGTGAAACCCCGTCTCTACTAAAAAT"
	"ACAAAAATTAGCCGGGCGTGGTGGCGCGCGCCTGTAATCCCA"
	"GCTACTCGGGAGGCTGAGGCAGGAGAATCGCTTGAACCCGGG"
	"AGGCGGAGGTTGCAGTGAGCCGAGATCGCGCCACTGCACTCC"
	"AGCCTGGGCGACAGAGCGAGACTCCGTCTCAAAAA"
};

/* "Strict SaferCPlusPlus" generally prefers the "lifespan aware" mse::mstd::array<> over mse::msearray<>,
but "lifespan safety" (i.e. premature deallocation) is not an issue for objects declared at global scope. */
mse::msearray<IUB, 15> iub =
{ { {
	{ 0.27f, 'a' },
	{ 0.12f, 'c' },
	{ 0.12f, 'g' },
	{ 0.27f, 't' },
	{ 0.02f, 'B' },
	{ 0.02f, 'D' },
	{ 0.02f, 'H' },
	{ 0.02f, 'K' },
	{ 0.02f, 'M' },
	{ 0.02f, 'N' },
	{ 0.02f, 'R' },
	{ 0.02f, 'S' },
	{ 0.02f, 'V' },
	{ 0.02f, 'W' },
	{ 0.02f, 'Y' }
	} } };

mse::msearray<IUB, 4> homosapiens =
{ { {
	{ 0.3029549426680f, 'a' },
	{ 0.1979883004921f, 'c' },
	{ 0.1975473066391f, 'g' },
	{ 0.3015094502008f, 't' }
	} } };

const int IM = 139968;
const float IM_RECIPROCAL = 1.0f / IM;

struct gen_random_state_type {
	int m_int = 42;
};

template<class gen_random_state_pointer_type>
uint32_t gen_random(gen_random_state_pointer_type state_ptr)
{
	static const int IA = 3877, IC = 29573;
	//static int last = 42;
	//last = (last * IA + IC) % IM;
	state_ptr->m_int = (state_ptr->m_int * IA + IC) % IM;
	return state_ptr->m_int;
}

char convert_trivial(char c)
{
	return c;
}

template<class iterator_type>
class repeat_generator_type {
public:
	using result_t = char;

	repeat_generator_type(iterator_type first, iterator_type last)
		: first(first), current(first), last(last)
	{ }
	template<class gen_random_state_pointer_type>
	result_t generate_random(gen_random_state_pointer_type unused_parameter)
	{
		if (current == last)
			current = first;
		iterator_type p = current;
		++current;
		return *p;
	}
private:
	iterator_type first;
	iterator_type current;
	iterator_type last;
};
template<class iterator_type>
repeat_generator_type<iterator_type>
make_repeat_generator(iterator_type first, iterator_type last)
{
	return repeat_generator_type<iterator_type>(first, last);
}

template<class iterator_type>
char convert_random(uint32_t random, iterator_type begin, iterator_type end)
{
	const float p = random * IM_RECIPROCAL;
	auto result = std::find_if(begin, end, [p](IUB i) { return p <= i.p; });
	return result->c;
}

char convert_IUB(uint32_t random)
{
	return convert_random(random, iub.cbegin(), iub.cend());
}

char convert_homosapiens(uint32_t random)
{
	return convert_random(random, homosapiens.cbegin(), homosapiens.cend());
}

template<class iterator_type>
class random_generator_type {
public:
	using result_t = uint32_t;

	random_generator_type(iterator_type first, iterator_type last)
		: first(first), last(last)
	{ }
	template<class gen_random_state_pointer_type>
	result_t generate_random(gen_random_state_pointer_type state_ptr)
	{
		return gen_random(state_ptr);
	}
private:
	iterator_type first;
	iterator_type last;
};
template<class iterator_type>
random_generator_type<iterator_type>
make_random_generator(iterator_type first, iterator_type last)
{
	return random_generator_type<iterator_type>(first, last);
}

template<class iterator_type>
void make_cumulative(iterator_type first, iterator_type last)
{
	std::partial_sum(first, last, first,
		[](IUB l, IUB r) -> IUB { r.p += l.p; return r; });
}

/* size_t is generally frowned upon due to it's problematic interaction with signed integers, but these
constants are just being used as declaration parameters for data types. */
const size_t CHARS_PER_LINE = 60;
const size_t CHARS_PER_LINE_INCL_NEWLINES = CHARS_PER_LINE + 1;
const size_t LINES_PER_BLOCK = 1024;
const size_t VALUES_PER_BLOCK = CHARS_PER_LINE * LINES_PER_BLOCK;
const size_t CHARS_PER_BLOCK_INCL_NEWLINES = CHARS_PER_LINE_INCL_NEWLINES * LINES_PER_BLOCK;

const mse::msear_size_t THREADS_TO_USE = std::max(1U, std::min(4U, std::thread::hardware_concurrency()));
//const mse::msear_size_t THREADS_TO_USE = 1U;

//define LOCK( mutex ) std::lock_guard< decltype( mutex ) > guard_{ mutex };

//std::mutex g_fillMutex;
//mse::msear_size_t g_fillThreadIndex = 0;
//mse::msear_size_t g_totalValuesToGenerate = 0;
struct fill_state_type {
	mse::msear_size_t m_fillThreadIndex = 0;
	mse::msear_size_t m_totalValuesToGenerate = 0;
	gen_random_state_type m_gen_random_state;
};
auto g_fill_state_access_requester = mse::make_asyncsharedobjectthatyouaresurehasnounprotectedmutablesreadwrite<fill_state_type>();

template<class iterator_type, class generator_pointer_type>
mse::msear_size_t fillBlock(mse::msear_size_t currentThread, iterator_type begin, generator_pointer_type generator_ptr)
{
	while (true)
	{
		//LOCK(g_fillMutex);
		auto fill_state_writelock_ptr = g_fill_state_access_requester.writelock_ptr();
		if (currentThread == fill_state_writelock_ptr->m_fillThreadIndex/*g_fillThreadIndex*/)
		{
			mse::TXScopeObj<fill_state_type> fill_state = *fill_state_writelock_ptr;
			// Select the next thread for this work.
			++(fill_state.m_fillThreadIndex);
			if (fill_state.m_fillThreadIndex >= THREADS_TO_USE)
			{
				fill_state.m_fillThreadIndex = 0;
			}

			// Do the work.
			const mse::msear_size_t valuesToGenerate = std::min(fill_state.m_totalValuesToGenerate, mse::msear_size_t(VALUES_PER_BLOCK));
			fill_state.m_totalValuesToGenerate -= valuesToGenerate;

			/* gen_random_state_xswfptr is a "safe" weak pointer to a member of an object declared on the stack. */
			auto gen_random_state_xswfptr = mse::make_xscopeweak(fill_state.m_gen_random_state, &fill_state);

			for (mse::msear_size_t valuesRemaining = 0; valuesRemaining < valuesToGenerate; ++valuesRemaining)
			{
				*begin = generator_ptr->generate_random(gen_random_state_xswfptr);
				++begin;
				//*begin++ = generator();
			}

			(*fill_state_writelock_ptr) = fill_state;
			return valuesToGenerate;
		}
	}
}

template<class BlockIter, class CharIter, class converter_type>
mse::msear_size_t convertBlock(BlockIter begin, BlockIter end, CharIter outCharacter, converter_type& converter)
{
	const auto beginCharacter = outCharacter;
	mse::msear_size_t col = 0;
	for (; begin != end; ++begin)
	{
		const uint32_t random = *begin;

		*outCharacter = converter(random);
		++outCharacter;
		if (++col >= CHARS_PER_LINE)
		{
			col = 0;
			*outCharacter = '\n';
			++outCharacter;
		}
	}
	//Check if we need to end the line
	if (0 != col)
	{
		//Last iteration didn't end the line, so finish the job.
		*outCharacter = '\n';
		++outCharacter;
	}

	return std::distance(beginCharacter, outCharacter);
}

//std::mutex g_outMutex;
//mse::msear_size_t g_outThreadIndex = std::numeric_limits<mse::msear_size_t>::max();
struct out_state_type
{
	mse::msear_size_t m_outThreadIndex = 0/*std::numeric_limits<mse::msear_size_t>::max()*/;
};
auto g_out_state_access_requester = mse::make_asyncsharedobjectthatyouaresurehasnounprotectedmutablesreadwrite<out_state_type>();

template<class char_array_type>
void writeCharacters(mse::msear_size_t currentThread, const char_array_type& char_array, mse::msear_size_t  count)
{
	while (true)
	{
		//LOCK(g_outMutex);
		auto out_state_writelock_ptr = g_out_state_access_requester.writelock_ptr();
		auto out_state = *out_state_writelock_ptr;
		if (out_state.m_outThreadIndex == std::numeric_limits<mse::msear_size_t>::max() || currentThread == out_state.m_outThreadIndex)
		{
			// Select the next thread for this work.
			out_state.m_outThreadIndex = currentThread + 1;
			if (out_state.m_outThreadIndex >= THREADS_TO_USE)
			{
				out_state.m_outThreadIndex = 0;
			}

			*out_state_writelock_ptr = out_state;

			// Do the work.
			if (count > char_array.size()) { std::cerr << "fatal error\n"; std::terminate(); }
			auto it = char_array.cbegin();
			for (mse::msear_size_t i = 0; i < count; i += 1) {
				std::cout.put(*it);
				++it;
			}
			//std::fwrite(char_array.data(), count, 1, stdout);
			return;
		}
	}
}


template<class generator_type, class generator_pointer_type, class converter_type>
void work(mse::msear_size_t currentThread, const generator_type&, generator_pointer_type generator_ptr, converter_type& converter)
{
	mse::mstd::array< typename generator_type::result_t, VALUES_PER_BLOCK > block;
	mse::mstd::array< char, CHARS_PER_BLOCK_INCL_NEWLINES > characters;

	while (true)
	{
		const auto bytesGenerated = fillBlock(currentThread, block.begin(), generator_ptr);

		if (bytesGenerated == 0)
		{
			break;
		}

		const auto charactersGenerated = convertBlock(block.begin(), block.begin() + bytesGenerated, characters.begin(), converter);

		writeCharacters(currentThread, characters, charactersGenerated);
	}
}

template <class generator_type, class generator_pointer_type, class converter_type >
void make(const std::string&/*const char* */desc, int n, const generator_type&, generator_pointer_type generator_ptr, converter_type converter) {
	std::cout << '>' << desc << '\n';

	{
		auto fill_state_writelock_ptr = g_fill_state_access_requester.writelock_ptr();
		fill_state_writelock_ptr->m_totalValuesToGenerate = n;
		fill_state_writelock_ptr->m_fillThreadIndex = 0;
		//g_totalValuesToGenerate = n;
		//g_fillThreadIndex = 0;
	}

	g_out_state_access_requester.writelock_ptr()->m_outThreadIndex = 0/*std::numeric_limits<mse::msear_size_t>::max()*/;

	mse::mstd::vector< std::thread > threads(THREADS_TO_USE - 1);
	for (mse::msear_size_t i = 0; i < threads.size(); ++i)
	{
		threads[i] = std::thread{ std::bind(&work< generator_type, generator_pointer_type, converter_type >, i, std::ref(*generator_ptr), generator_ptr, std::ref(converter)) };
	}

	work(threads.size(), *generator_ptr, generator_ptr, converter);

	for (auto& thread : threads)
	{
		thread.join();
	}
}

int main(int argc, char *argv[])
{
	mse::CInt n = 1000;
	if (argc < 2 || (n = std::stoi(argv[1])) <= 0) {
		std::cerr << "usage: " << argv[0] << " length\n";
		return 1;
	}

	make_cumulative(iub.begin(), iub.end());
	make_cumulative(homosapiens.begin(), homosapiens.end());

	{
		mse::TXScopeObj<repeat_generator_type<std::string::const_iterator>> repeat_generator1 = make_repeat_generator(alu.begin(), alu.end());
		make("ONE Homo sapiens alu", n * 2,
			repeat_generator1, &repeat_generator1,
			&convert_trivial);
	}
	{
		mse::TXScopeObj<random_generator_type<mse::msearray<IUB, 15>::const_iterator>> random_generator1 = make_random_generator(iub.cbegin(), iub.cend());
		make("TWO IUB ambiguity codes", n * 3,
			random_generator1, &random_generator1,
			&convert_IUB);
	}
	{
		mse::TXScopeObj<random_generator_type<mse::msearray<IUB, 4>::const_iterator>> random_generator2 = make_random_generator(homosapiens.cbegin(), homosapiens.cend());
		make("THREE Homo sapiens frequency", n * 5,
			random_generator2, &random_generator2,
			&convert_homosapiens);
	}
	return 0;
}
