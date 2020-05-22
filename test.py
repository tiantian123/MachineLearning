#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author: Tian Chen
# @times: 2020/4/13  14:34
# @File: test.py
# @email: chentianfighting@126.com
import asyncio
import os
import psutil
import time
import re
from typing import List


# 显示当前python程序占用的内存大小
def show_mem_info(hint):
    pid = os.getpid()
    p = psutil.Process(pid)

    info = p.memory_full_info()
    memory = info.uss / 1024. / 1024
    print(f'{hint} memory used: {memory} MB')


def is_subsequence(a, b):
    b = iter(b)
    # gen = (i for i in a)
    # gen = ((i in b) for i in a)
    return all(((i in b) for i in a))

# print(is_subsequence([1, 3, 5], [1, 2, 3, 4, 5]))
# print(is_subsequence([1, 4, 3], [1, 2, 3, 4, 5]))


class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        if not grid or not grid[0]: return 0
        m, n = len(grid), len(grid[0])
        uf = UnionFind(grid)
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1':
                    uf.Union(i, j)
        return uf.count


class UnionFind(object):
    def __init__(self, grid: List[List[str]]):
        m, n = len(grid), len(grid[0])
        self.count = 0
        self.parent = [-1] * (m*n)
        self.rank = [0] * (m*n)
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1':
                    self.parent[i*n + j] = i*n + j
                    self.count += 1

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
        return self.parent[x]

    def Union(self, x, y):
        rootx = self.parent[x]
        rooty = self.parent[y]

        if rootx != rooty:
            if self.rank[rootx] < self.rank[rooty]:
                self.parent[rootx] = rooty
            elif self.rank[rootx] > self.rank[rooty]:
                self.parent[rooty] = rootx
            else:
                self.parent[rootx] = rooty
                self.rank[rooty] += 1
            self.count -= 1


async def factorial(name, number):
    f = 1
    for i in range(2, number + 1):
        print(f"Task {name}: Compute factorial({i})")
        await asyncio.sleep(1)
        f *= i
    print(f"Task {name}: factorial({number}) = {f}")

async def main():
    # schedule three calls *concurrently*:
    await asyncio.gather(
        factorial("A", 2),
        factorial("B", 3),
        factorial("C", 4),
    )

asyncio.run(main())
