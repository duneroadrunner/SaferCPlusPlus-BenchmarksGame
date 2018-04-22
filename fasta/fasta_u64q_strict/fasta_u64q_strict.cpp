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

#define MSE_MSEARRAY_USE_MSE_PRIMITIVES
#define MSE_MSEVECTOR_USE_MSE_PRIMITIVES

#ifdef _MSC_VER
#pragma warning( push )  
#pragma warning( disable : 4503 )
#endif /*_MSC_VER*/

#include <string>
#include <limits>
#include "mseprimitives.h"
#include "msemstdarray.h"
#include "msemstdvector.h"
#include "mseasyncshared.h"
#include "msescope.h"
#include "msepoly.h"
#include "msealgorithm.h"

struct IUB
{
	float p;
	char c;
};
/* ShareableIUB is just a version of IUB that is declared to meet the criteria for safe sharing between threads (basically
no pointer/reference (or mutable) members). */
typedef mse::us::TUserDeclaredAsyncShareableObj<IUB> ShareableIUB;

const mse::TXScopeObj<mse::nii_string> alu =
{
	"GGCCGGGCGCGGTGGCTCACGCCTGTAATCCCAGCACTTTGG"
	"GAGGCCGAGGCGGGCGGATCACCTGAGGTCAGGAGTTCGAGA"
	"CCAGCCTGGCCAACATGGTGAAACCCCGTCTCTACTAAAAAT"
	"ACAAAAATTAGCCGGGCGTGGTGGCGCGCGCCTGTAATCCCA"
	"GCTACTCGGGAGGCTGAGGCAGGAGAATCGCTTGAACCCGGG"
	"AGGCGGAGGTTGCAGTGAGCCGAGATCGCGCCACTGCACTCC"
	"AGCCTGGGCGACAGAGCGAGACTCCGTCTCAAAAA"
};
auto g_alu_shimptr = mse::make_asyncsharedv2immutable<mse::nii_string>(alu);

template<class iterator_type>
void make_cumulative(iterator_type first, iterator_type last)
{
	std::partial_sum(first, last, first,
		[](IUB l, IUB r) -> IUB { r.p += l.p; return r; });
}

const int IM = 139968;
const float IM_RECIPROCAL = 1.0f / IM;

struct gen_random_state_type {
	int m_int = 42;
};

template<class gen_random_state_pointer_type>
uint32_t gen_random(gen_random_state_pointer_type state_ptr)
{
	static const int IA = 3877, IC = 29573;
	/* The static int was being accessed and modified from multiple asynchronous threads, so we moved it into
	a separate "state" structure so that we could use SaferCPlusPlus' automatic access controls to ensure safe
	access (i.e. no race conditions).*/
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
	mse::TMemberObj<iterator_type> first;
	mse::TMemberObj<iterator_type> current;
	mse::TMemberObj<iterator_type> last;
};
template<class iterator_type>
repeat_generator_type<iterator_type>
make_repeat_generator(iterator_type first, iterator_type last)
{
	return repeat_generator_type<iterator_type>(first, last);
}

template<class array_ptr_t>
char convert_random(uint32_t random, const array_ptr_t& data_array_ptr)
{
	const float p = random * IM_RECIPROCAL;
	auto result = mse::xscope_ra_const_find_if(data_array_ptr, [p](IUB i) { return p <= i.p; }).value();
	return result->c;
}

template<class iterator_type>
class random_generator_type {
public:
	using result_t = uint32_t;

	random_generator_type(iterator_type first, iterator_type last)
		//: first(first), last(last)
	{ }
	template<class gen_random_state_pointer_type>
	result_t generate_random(gen_random_state_pointer_type state_ptr)
	{
		return gen_random(state_ptr);
	}
private:
	//mse::TMemberObj<iterator_type> first;
	//mse::TMemberObj<iterator_type> last;
};
template<class iterator_type>
random_generator_type<iterator_type>
make_random_generator(iterator_type first, iterator_type last)
{
	return random_generator_type<iterator_type>(first, last);
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
		/* Usually each thread would hold their own copy of the "access requester", but if the original
		access requester happens to be global, then each thread can just use the global one (without fear
		that it might be deallocated before it's finished using it). */
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

			/* Obtaining a scope pointer to fill_state.m_gen_random_state. */
			auto gen_random_state_xscpptr = mse::make_xscope_pointer_to_member_v2(&fill_state, &fill_state_type::m_gen_random_state);

			for (mse::msear_size_t valuesRemaining = 0; valuesRemaining < valuesToGenerate; ++valuesRemaining)
			{
				*begin = generator_ptr->generate_random(gen_random_state_xscpptr);
				++begin;
				//*begin++ = generator();
			}

			(*fill_state_writelock_ptr) = fill_state;
			return valuesToGenerate;
		}
	}
}

template<class BlockIter, class CharIter, class converter_function_type, class converter_data_xscope_pointer_type>
mse::msear_size_t convertBlock(BlockIter begin, BlockIter end, CharIter outCharacter, converter_function_type convert, converter_data_xscope_pointer_type converter_data_xscpcptr)
{
	auto xscope_data_begin_citer = mse::make_xscope_const_iterator(converter_data_xscpcptr);
	auto xscope_data_end_citer = xscope_data_begin_citer + mse::as_a_size_t((*converter_data_xscpcptr).size());

	const auto beginCharacter = outCharacter;
	mse::msear_size_t col = 0;
	for (; begin != end; ++begin)
	{
		const uint32_t random = *begin;

		*outCharacter = convert(random, converter_data_xscpcptr);
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
typedef mse::us::TUserDeclaredAsyncShareableObj<out_state_type> shareable_out_state_type;
auto g_out_state_access_requester = mse::make_asyncsharedv2readwrite<shareable_out_state_type>();

template<class char_array_pointer_type>
void writeCharacters(mse::msear_size_t currentThread, char_array_pointer_type char_array_pointer, mse::msear_size_t count)
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

//define DISABLE_OUTPUT
#ifndef DISABLE_OUTPUT

			// Do the work.
			(*char_array_pointer).write_bytes(std::cout, count);
			//std::fwrite((*char_array_pointer).data(), count, 1, stdout);

#endif // !DISABLE_OUTPUT

			return;
		}
	}
}

template<class generator_make_function_type, class converter_function_type, class converter_data_type>
void work(mse::msear_size_t currentThread, generator_make_function_type generator_make_function, converter_function_type convert, converter_data_type converter_data)
{
	mse::TXScopeObj<converter_data_type> xscope_converter_data = converter_data;

	typedef decltype(generator_make_function()) generator_type;
	mse::TXScopeObj<generator_type> generator = generator_make_function();
	auto generator_xscpptr = &generator;

	mse::TXScopeObj<mse::nii_array< typename generator_type::result_t, VALUES_PER_BLOCK > > block;
	mse::TXScopeObj<mse::nii_array< char, CHARS_PER_BLOCK_INCL_NEWLINES > > characters;

	while (true)
	{
		auto block_begin_iter = mse::make_xscope_iterator(&block);
		const auto bytesGenerated = fillBlock(currentThread, block_begin_iter, generator_xscpptr);

		if (bytesGenerated == 0)
		{
			break;
		}

		auto block_begin_iter2 = mse::make_xscope_iterator(&block);
		auto characters_begin_iter = mse::make_xscope_iterator(&characters);
		const auto charactersGenerated = convertBlock(block_begin_iter2, block_begin_iter2 + bytesGenerated, characters_begin_iter, convert, &xscope_converter_data);

		writeCharacters(currentThread, &characters, charactersGenerated);
	}
}

template<class generator_make_function_type, class converter_function_type, class converter_data_type>
void make(const mse::nii_string/*const char* */desc, mse::CInt n, generator_make_function_type generator_make_function, converter_function_type convert, converter_data_type converter_data) {
	std::cout << '>' << desc << '\n';

	{
		auto fill_state_writelock_ptr = g_fill_state_access_requester.writelock_ptr();
		fill_state_writelock_ptr->m_totalValuesToGenerate = n;
		fill_state_writelock_ptr->m_fillThreadIndex = 0;
		//g_totalValuesToGenerate = n;
		//g_fillThreadIndex = 0;
	}

	g_out_state_access_requester.writelock_ptr()->m_outThreadIndex = 0/*std::numeric_limits<mse::msear_size_t>::max()*/;

	mse::mstd::vector< mse::mstd::thread > threads(THREADS_TO_USE - 1);
	for (mse::msear_size_t i = 0; i < threads.size(); ++i)
	{
		threads[i] = mse::mstd::thread(work< generator_make_function_type, converter_function_type, converter_data_type >, mse::as_a_size_t(i), generator_make_function, convert, converter_data);
	}

	work(threads.size(), generator_make_function, convert, converter_data);

	for (auto& thread : threads)
	{
		thread.join();
	}
}

auto make_alu_repeat_generator() {
	return make_repeat_generator(g_alu_shimptr->ss_cbegin(g_alu_shimptr), g_alu_shimptr->ss_cend(g_alu_shimptr));
}
auto make_iub_random_generator() {
	return make_random_generator(nullptr, nullptr);
}
auto make_homosapiens_random_generator() {
	return make_random_generator(nullptr, nullptr);
}

int main(int argc, char *argv[])
{
	mse::CInt n = 1000;
	if (argc < 2 || (n = std::stoi(argv[1])) <= 0) {
		std::cerr << "usage: " << argv[0] << " length\n";
		return 1;
	}

	/* Non-const globals are considered unsafe, so these were made non-global. */
	mse::TXScopeObj<mse::nii_array<IUB, 15> > iub0 = mse::nii_array<IUB, 15>
	{ {
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
		} };
	mse::TXScopeObj<mse::nii_array<IUB, 4> > homosapiens0 = mse::nii_array<IUB, 4>
	{ {
		{ 0.3029549426680f, 'a' },
	{ 0.1979883004921f, 'c' },
	{ 0.1975473066391f, 'g' },
	{ 0.3015094502008f, 't' }
		} };

	auto iub0_begin = mse::make_xscope_iterator(&iub0);
	auto iub0_end = iub0_begin + iub0.size();
	make_cumulative(iub0_begin, iub0_end);

	auto homosapiens0_begin = mse::make_xscope_iterator(&homosapiens0);
	auto homosapiens0_end = homosapiens0_begin + homosapiens0.size();
	make_cumulative(homosapiens0_begin, homosapiens0_end);

	/* Because we're going to pass and/or share these arrays between threads we need copies that are recognized (by the
	library) as safely shareable. */
	typedef mse::nii_array<ShareableIUB, 15> shareable_iub_array_t;
	mse::TXScopeObj<shareable_iub_array_t> iub;
	for (mse::msear_size_t i = 0; i < iub0.size(); i += 1) {
		iub[i] = iub0[i];
	}
	typedef mse::nii_array<ShareableIUB, 4> shareable_homosapiens_array_t;
	mse::TXScopeObj<shareable_homosapiens_array_t> homosapiens;
	for (mse::msear_size_t i = 0; i < homosapiens0.size(); i += 1) {
		homosapiens[i] = homosapiens0[i];
	}

	struct functions {
		typedef mse::TXScopeFixedPointer<mse::nii_string> alu_xscope_pointer_type;
		static char convert_trivial(char c, alu_xscope_pointer_type) {
			return c;
		}
		typedef mse::TXScopeFixedPointer<shareable_iub_array_t> iub_xscope_pointer_type;
		static char convert_IUB(uint32_t random, iub_xscope_pointer_type iub_xscope_pointer) {
			return convert_random(random, iub_xscope_pointer);
		}
		typedef mse::TXScopeFixedPointer<shareable_homosapiens_array_t> homosapiens_xscope_pointer_type;
		static char convert_homosapiens(uint32_t random, homosapiens_xscope_pointer_type homosapiens_xscope_pointer) {
			return convert_random(random, homosapiens_xscope_pointer);
		}
	};

	{
		make("ONE Homo sapiens alu", n * 2,
			make_alu_repeat_generator,
			functions::convert_trivial, mse::nii_string("placeholder data"));
	}
	{
		make("TWO IUB ambiguity codes", n * 3,
			make_iub_random_generator,
			functions::convert_IUB, shareable_iub_array_t(iub));
	}
	{
		make("THREE Homo sapiens frequency", n * 5,
			make_homosapiens_random_generator,
			functions::convert_homosapiens, shareable_homosapiens_array_t(homosapiens));
	}
	return 0;
}

#ifdef _MSC_VER
#pragma warning( pop )  
#endif /*_MSC_VER*/
