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

#include <string>
#include <limits>
#include "msemsearray.h"
#include "msemsevector.h"
#include "mseasyncshared.h"

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

uint32_t gen_random(gen_random_state_type& state)
{
	static const int IA = 3877, IC = 29573;
	//static int last = 42;
	auto& last = state.m_int;
	last = (last * IA + IC) % IM;
	return last;
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
	result_t generate_random(gen_random_state_type& unused_parameter)
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
	return convert_random(random, iub.ss_cbegin(), iub.ss_cend());
}

char convert_homosapiens(uint32_t random)
{
	return convert_random(random, homosapiens.ss_cbegin(), homosapiens.ss_cend());
}

template<class iterator_type>
class random_generator_type {
public:
	using result_t = uint32_t;

	random_generator_type(iterator_type first, iterator_type last)
		: first(first), last(last)
	{ }
	result_t generate_random(gen_random_state_type& state)
	{
		return gen_random(state);
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

template<class iterator_type, class generator_type>
mse::msear_size_t fillBlock(mse::msear_size_t currentThread, iterator_type begin, generator_type& generator)
{
	while (true)
	{
		//LOCK(g_fillMutex);
		auto fill_state_writelock_ptr = g_fill_state_access_requester.writelock_ptr();
		if (currentThread == fill_state_writelock_ptr->m_fillThreadIndex/*g_fillThreadIndex*/)
		{
			auto fill_state = *fill_state_writelock_ptr;
			// Select the next thread for this work.
			++(fill_state.m_fillThreadIndex);
			if (fill_state.m_fillThreadIndex >= THREADS_TO_USE)
			{
				fill_state.m_fillThreadIndex = 0;
			}

			// Do the work.
			const mse::msear_size_t valuesToGenerate = std::min(fill_state.m_totalValuesToGenerate, VALUES_PER_BLOCK);
			fill_state.m_totalValuesToGenerate -= valuesToGenerate;

			for (mse::msear_size_t valuesRemaining = 0; valuesRemaining < valuesToGenerate; ++valuesRemaining)
			{
				*begin = generator.generate_random(fill_state.m_gen_random_state);
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
void writeCharacters(mse::msear_size_t currentThread, const char_array_type& char_array, size_t count)
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


template<class generator_type, class converter_type>
void work(mse::msear_size_t currentThread, generator_type& generator, converter_type& converter)
{
	mse::msearray< typename generator_type::result_t, VALUES_PER_BLOCK > block;
	mse::msearray< char, CHARS_PER_BLOCK_INCL_NEWLINES > characters;

	while (true)
	{
		const auto bytesGenerated = fillBlock(currentThread, block.ss_begin(), generator);

		if (bytesGenerated == 0)
		{
			break;
		}

		const auto charactersGenerated = convertBlock(block.ss_begin(), block.ss_begin() + bytesGenerated, characters.ss_begin(), converter);

		writeCharacters(currentThread, characters, charactersGenerated);
	}
}

template <class generator_type, class converter_type >
void make(const std::string&/*const char* */desc, int n, generator_type generator, converter_type converter) {
	std::cout << '>' << desc << '\n';

	{
		auto fill_state_writelock_ptr = g_fill_state_access_requester.writelock_ptr();
		fill_state_writelock_ptr->m_totalValuesToGenerate = n;
		fill_state_writelock_ptr->m_fillThreadIndex = 0;
		//g_totalValuesToGenerate = n;
		//g_fillThreadIndex = 0;
	}

	g_out_state_access_requester.writelock_ptr()->m_outThreadIndex = 0/*std::numeric_limits<mse::msear_size_t>::max()*/;

	mse::msevector< std::thread > threads(THREADS_TO_USE - 1);
	for (mse::msear_size_t i = 0; i < threads.size(); ++i)
	{
		threads[i] = std::thread{ std::bind(&work< generator_type, converter_type >, i, std::ref(generator), std::ref(converter)) };
	}

	work(threads.size(), generator, converter);

	for (auto& thread : threads)
	{
		thread.join();
	}
}

int main(int argc, char *argv[])
{
	int n = 1000;
	if (argc < 2 || (n = std::stoi(argv[1])) <= 0) {
		std::cerr << "usage: " << argv[0] << " length\n";
		return 1;
	}

	make_cumulative(iub.ss_begin(), iub.ss_end());
	make_cumulative(homosapiens.ss_begin(), homosapiens.ss_end());

	make("ONE Homo sapiens alu", n * 2,
		make_repeat_generator(alu.begin(), alu.end()),
		&convert_trivial);
	make("TWO IUB ambiguity codes", n * 3,
		make_random_generator(iub.ss_cbegin(), iub.ss_cend()),
		&convert_IUB);
	make("THREE Homo sapiens frequency", n * 5,
		make_random_generator(homosapiens.ss_cbegin(), homosapiens.ss_cend()),
		&convert_homosapiens);
	return 0;
}
