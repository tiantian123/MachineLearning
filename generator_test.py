#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author: Tian Chen
# @times: 2020/4/8  10:43
# @File: generator_test.py
# @email: chentianfighting@126.com
import asyncio
import threading


@asyncio.coroutine
def hello():
    print(f'hello world! {threading.currentThread()}')
    r = yield from asyncio.sleep(3)
    print(f'hello again! {threading.currentThread()}')


@asyncio.coroutine
def wget(host):
    print(f'wget {host}...')
    connect = asyncio.open_connection(host, 80)
    reader, writer  = yield from connect
    header = f'Get / http/1.0\r\nHost:{host}\r\n\r\n'
    writer.write(header.encode('utf-8'))
    yield from writer.drain()
    while True:
        line = yield from reader.readline()
        if line == b'\r\n':
            break
        print(f'{host} header > {line.decode("utf-8").rstrip()}')
    writer.close()


loop = asyncio.get_event_loop()
tasks = [wget(host) for host in ['www.sina.com.cn', 'www.sohu.com', 'www.163.com']]
loop.run_until_complete(asyncio.wait(tasks))
loop.close()