/* The Computer Language Benchmarks Game
http://benchmarksgame.alioth.debian.org/
contributed by David Turnbull
modified by Pascal Urban (parallel creation of binary trees using libdispatch)
modified by Maurus Kühne (checkTree uses inout parameters)
modified for Swift 3.0 by Daniel Muellenborn
*/

#ifdef _MSC_VER
#include "windows.h"
#undef max
#undef min
#endif // _MSC_VER

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <algorithm>
#include <vector>
#include <string>
//include <omp.h>
#include <future>

#include "mseprimitives.h"
#define MSE_MSEVECTOR_USE_MSE_PRIMITIVES 1
#include "msemstdvector.h"
#include "msescope.h"
#include "mseasyncshared.h"

struct TreeNodeItem {
	TreeNodeItem() {}
	TreeNodeItem(mse::CInt item_param) : item(item_param) {}
	TreeNodeItem(mse::msev_size_t left_param, mse::msev_size_t right_param, mse::CInt item_param) : left(left_param), right(right_param), item(item_param), is_not_leaf(true) {}
	mse::msev_size_t left = 0;
	mse::msev_size_t right = 0;
	mse::CInt item = 0;
	mse::CBool is_not_leaf = false;
};

template <class _TNodePoolPointer>
mse::msev_size_t buildTree(_TNodePoolPointer t_ptr, mse::CInt item, mse::CInt depth) {
	if (depth > 0) {
		(*t_ptr).emplace_back(
			TreeNodeItem{
			buildTree(t_ptr, 2 * item - 1, depth - 1),
			buildTree(t_ptr, 2 * item, depth - 1),
			item });
		return mse::as_a_size_t((*t_ptr).size()) - 1;
	}
	else {
		(*t_ptr).emplace_back(TreeNodeItem{ item });
		return mse::as_a_size_t((*t_ptr).size()) - 1;
	}
}

template <class _TNodePoolPointer>
mse::CInt checkTree(_TNodePoolPointer t_ptr, mse::msev_size_t i) {
	const TreeNodeItem tni = (*t_ptr)[i];
	if (tni.is_not_leaf) {
		return tni.item + checkTree(t_ptr, tni.left) - checkTree(t_ptr, tni.right);
	}
	else {
		return 0;
	}
}

int GetThreadCount()
{
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
	for (int i = 0; i < 8; i++)
	{
		if (CPU_ISSET(i, &cs))
			count++;
	}
	return count;
#endif // _MSC_VER
}

auto foo1(mse::CInt depthIteration, const mse::CInt minDepth, const mse::CInt maxDepth) {
	mse::TXScopeObj<mse::nii_vector<TreeNodeItem>> shortLivedPool;
	auto depth = minDepth + depthIteration * 2;

	auto iterations = 1 << (maxDepth - depth + minDepth);
	mse::CInt check = 0;
	for (auto i = 1; i <= iterations; i += 1) {
		shortLivedPool.resize(0);
		auto t1 = buildTree(&shortLivedPool, i, depth);
		check += checkTree(&shortLivedPool, t1);
		shortLivedPool.resize(0);
		auto t2 = buildTree(&shortLivedPool, -i, depth);
		check += checkTree(&shortLivedPool, t2);
	}
	//checksumOutput[depthIteration] = check;
	return check;
}

int main(int argc, char *argv[])
{
	mse::CInt n = 10;
	if (2 <= argc) {
		n = std::stoi(argv[1]);
	}

	mse::CInt minDepth = 4;
	mse::CInt maxDepth = n;
	mse::CInt stretchDepth = n + 1;

	mse::TXScopeObj<mse::nii_vector<TreeNodeItem>> longLivedPool;

	auto longLivedTree = buildTree(&longLivedPool, 0, stretchDepth);

	auto chk = checkTree(&longLivedPool, longLivedTree);
	std::cout << "stretch tree of depth " << stretchDepth << "\t "
		<< "check: " << chk << std::endl;

	longLivedPool.resize(0);
	longLivedTree = buildTree(&longLivedPool, 0, maxDepth);

	const auto numberOfIterations = (maxDepth - minDepth) / 2 + 1;

	mse::nii_vector<mse::CInt> checksumOutput(numberOfIterations);

	mse::mstd::vector<std::future<mse::CInt>> futures;
	for (auto depthIteration = 0; depthIteration < numberOfIterations; depthIteration += 1) {
		futures.emplace_back(mse::mstd::async(foo1, depthIteration, minDepth, maxDepth));
	}
	int depthIteration = 0;
	for (auto it = futures.begin(); futures.end() != it; it++, depthIteration++) {
		checksumOutput[depthIteration] = (*it).get();
	}


	const auto checksumOutput_size = checksumOutput.size();
	for (mse::msev_size_t depthIteration = 0; depthIteration < checksumOutput_size; depthIteration += 1) {
		auto depth = minDepth + depthIteration * 2;
		auto iterations = 1 << (maxDepth - depth + minDepth);
		auto check = checksumOutput[depthIteration];

		std::cout << (iterations * 2) << "\t trees of depth " << (depth) << "\t check: " << (check) << std::endl;
	}

	std::cout << "long lived tree of depth " << maxDepth << "\t check: " << (checkTree(&longLivedPool, longLivedTree)) << std::endl;
}
