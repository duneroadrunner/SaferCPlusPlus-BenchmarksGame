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
#include <omp.h>
#include <algorithm>
#include <vector>
#include <string>
#include <omp.h>

#include "mseprimitives.h"
#include "msemsevector.h"

struct TreeNodeItem {
	TreeNodeItem() {}
	TreeNodeItem(mse::msev_int left_param, mse::msev_int right_param, mse::msev_int item_param) : left(left_param), right(right_param), item(item_param) {}
	mse::msev_int left = 0;
	mse::msev_int right = 0;
	mse::CInt item = 0;
};

mse::CInt buildTree(mse::msevector<TreeNodeItem> &t, mse::CInt item, mse::CInt depth) {
	if (depth > 0) {
		t.emplace_back(
			TreeNodeItem{
			buildTree(t, 2 * item - 1, depth - 1),
			buildTree(t, 2 * item, depth - 1),
			item });
		return mse::CInt(t.size()) - 1;
	}
	else {
		t.emplace_back(TreeNodeItem{ -1, -1, item });
		return mse::CInt(t.size()) - 1;
	}
}

mse::msev_int checkTree(const mse::msevector<TreeNodeItem> &t, mse::msev_int i) {
	const TreeNodeItem& ti = t[i];
	if (ti.left < 0) {
		return 0;
	}
	else {
		return ti.item + checkTree(t, ti.left) - checkTree(t, ti.right);
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

int main(int argc, char *argv[])
{
	mse::CInt n = 10;
	if (2 <= argc) {
		n = std::stoi(argv[1]);
	}

	mse::CInt minDepth = 4;
	mse::CInt maxDepth = n;
	mse::CInt stretchDepth = n + 1;

	mse::msevector<TreeNodeItem> longLivedPool;

	auto longLivedTree = buildTree(longLivedPool, 0, stretchDepth);

	auto chk = checkTree(longLivedPool, longLivedTree);
	std::cout << "stretch tree of depth " << stretchDepth << "\t "
		<< "check: " << chk << std::endl;

	longLivedPool.resize(0);
	longLivedTree = buildTree(longLivedPool, 0, maxDepth);

	const auto numberOfIterations = (maxDepth - minDepth) / 2 + 1;

	mse::msevector<mse::CInt> checksumOutput(numberOfIterations);

#pragma omp parallel for default(shared) num_threads(GetThreadCount()) schedule(dynamic, 1)
	for (auto depthIteration = 0; depthIteration < int(numberOfIterations)/*omp requires this be an actual int*/; depthIteration += 1) {
		mse::msevector<TreeNodeItem> shortLivedPool;
		auto depth = minDepth + depthIteration * 2;

		auto iterations = 1 << (maxDepth - depth + minDepth);
		auto check = 0;
		for (auto i = 1; i <= iterations; i += 1) {
			shortLivedPool.resize(0);
			auto t1 = buildTree(shortLivedPool, i, depth);
			check += checkTree(shortLivedPool, t1);
			shortLivedPool.resize(0);
			auto t2 = buildTree(shortLivedPool, -i, depth);
			check += checkTree(shortLivedPool, t2);
		}
		checksumOutput[depthIteration] = check;
	}

	const mse::msev_int checksumOutput_size = checksumOutput.size();
	for (auto depthIteration = 0; depthIteration < checksumOutput_size; depthIteration += 1) {
		auto depth = minDepth + depthIteration * 2;
		auto iterations = 1 << (maxDepth - depth + minDepth);
		auto check = checksumOutput[depthIteration];

		std::cout << (iterations * 2) << "\t trees of depth " << (depth) << "\t check: " << (check) << std::endl;
	}

	std::cout << "long lived tree of depth " << maxDepth << "\t check: " << (checkTree(longLivedPool, longLivedTree)) << std::endl;
}
