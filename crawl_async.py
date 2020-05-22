#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author: Tian Chen
# @times: 2020/4/8  9:17
# @File: crawl_async.py
# @email: chentianfighting@126.com
import asyncio
import time
import random


async def consumer(queue, id):
    while True:
        val = await queue.get()
        print(f'{id} get a val {val}')
        await asyncio.sleep(1)

async def producer(queue, id):
    for i in range(5):
        val = random.randint(1,10)
        await queue.put(val)
        print(f'{id} put a val {val}')
        await asyncio.sleep(1)

async def main():
    queue = asyncio.Queue()

    consumer_1 = asyncio.create_task(consumer(queue, 'consumer_1'))
    consumer_2 = asyncio.create_task(consumer(queue, 'consumer_2'))

    producer_1 = asyncio.create_task(producer(queue, 'producer_1'))
    producer_2 = asyncio.create_task(producer(queue, 'producer_2'))

    await asyncio.sleep(10)
    consumer_1.cancel()
    consumer_2.cancel()

    await asyncio.gather(consumer_1,consumer_2, producer_1, producer_2, return_exceptions=True)

start = time.time()
asyncio.run(main())
end = time.time()
print(f'runing time {round(end - start, 2)}')