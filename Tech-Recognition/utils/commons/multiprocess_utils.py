import os
import traceback
from functools import partial
from tqdm import tqdm
import torch

def chunked_worker(worker_id, args_queue=None, results_queue=None, init_ctx_func=None):
    ctx = init_ctx_func(worker_id) if init_ctx_func is not None else None
    while True:
        args = args_queue.get()
        if args == '<KILL>':
            return
        job_idx, map_func, arg = args
        try:
            map_func_ = partial(map_func, ctx=ctx) if ctx is not None else map_func
            if isinstance(arg, dict):
                res = map_func_(**arg)
            elif isinstance(arg, (list, tuple)):
                res = map_func_(*arg)
            else:
                res = map_func_(arg)
            results_queue.put((job_idx, res))
        except:
            traceback.print_exc()
            results_queue.put((job_idx, None))


class MultiprocessManager:
    def __init__(self, num_workers=None, init_ctx_func=None, multithread=False, queue_max=-1):
        if multithread:
            from multiprocessing.dummy import Queue, Process
        else:
            from multiprocessing import Queue, Process
        if num_workers is None:
            num_workers = int(os.getenv('N_PROC', os.cpu_count()))
        self.num_workers = num_workers
        self.results_queue = Queue(maxsize=-1)
        self.jobs_pending = []
        self.args_queue = Queue(maxsize=queue_max)
        self.workers = []
        self.total_jobs = 0
        self.multithread = multithread
        for i in range(num_workers):
            if multithread:
                p = Process(target=chunked_worker,
                            args=(i, self.args_queue, self.results_queue, init_ctx_func))
            else:
                p = Process(target=chunked_worker,
                            args=(i, self.args_queue, self.results_queue, init_ctx_func),
                            daemon=True)
            self.workers.append(p)
            p.start()

    def add_job(self, func, args):
        if not self.args_queue.full():
            self.args_queue.put((self.total_jobs, func, args))
        else:
            self.jobs_pending.append((self.total_jobs, func, args))
        self.total_jobs += 1

    def get_results(self):
        self.n_finished = 0
        while self.n_finished < self.total_jobs:
            while len(self.jobs_pending) > 0 and not self.args_queue.full():
                self.args_queue.put(self.jobs_pending[0])
                self.jobs_pending = self.jobs_pending[1:]
            job_id, res = self.results_queue.get()
            yield job_id, res
            self.n_finished += 1
        for w in range(self.num_workers):
            self.args_queue.put("<KILL>")
        for w in self.workers:
            w.join()

    def close(self):
        if not self.multithread:
            for w in self.workers:
                w.terminate()

    def __len__(self):
        return self.total_jobs


def multiprocess_run_tqdm(map_func, args, num_workers=None, ordered=True, init_ctx_func=None,
                          multithread=False, queue_max=-1, desc=None):
    for i, res in tqdm(
            multiprocess_run(map_func, args, num_workers, ordered, init_ctx_func, multithread,
                             queue_max=queue_max),
            total=len(args), desc=desc):
        yield i, res


def multiprocess_run(map_func, args, num_workers=None, ordered=True, init_ctx_func=None, multithread=False,
                     queue_max=-1):
    """
    Multiprocessing running chunked jobs.

    Examples:
    >>> for res in tqdm(multiprocess_run(job_func, args):
    >>>     print(res)

    :param map_func:
    :param args:
    :param num_workers:
    :param ordered:
    :param init_ctx_func:
    :param q_max_size:
    :param multithread:
    :return:
    """
    if num_workers is None:
        num_workers = int(os.getenv('N_PROC', os.cpu_count()))
    manager = MultiprocessManager(num_workers, init_ctx_func, multithread, queue_max=queue_max)
    for arg in args:
        manager.add_job(map_func, arg)
    if ordered:
        n_jobs = len(args)
        results = ['<WAIT>' for _ in range(n_jobs)]
        i_now = 0
        for job_i, res in manager.get_results():
            results[job_i] = res
            while i_now < n_jobs and (not isinstance(results[i_now], str) or results[i_now] != '<WAIT>'):
                yield i_now, results[i_now]
                results[i_now] = None
                i_now += 1
    else:
        for job_i, res in manager.get_results():
            yield job_i, res
    manager.close()


# #### this is for multiprocessing on cuda
# this doesn't work
import platform
import re
import traceback
from torch.multiprocessing import Manager, Process, current_process, get_context

is_main_process = not bool(re.match(r'((.*Process)|(SyncManager)|(.*PoolWorker))-\d+', current_process().name))

def main_process_print(self, *args, sep=' ', end='\n', file=None):
    if is_main_process:
        print(self, *args, sep=sep, end=end, file=file)

def chunked_worker_run(map_func, args, results_queue=None):
    for a in args:
        # noinspection PyBroadException
        try:
            res = map_func(*a)
            results_queue.put(res)
        except KeyboardInterrupt:
            break
        except Exception:
            traceback.print_exc()
            results_queue.put(None)

def multiprocess_run_cuda(map_func, args, num_workers, q_max_size=1000):
    num_jobs = len(args)
    if num_jobs < num_workers:
        num_workers = num_jobs

    queues = [Manager().Queue(maxsize=q_max_size // num_workers) for _ in range(num_workers)]
    if platform.system().lower() != 'windows':
        process_creation_func = get_context('spawn').Process
    else:
        process_creation_func = Process

    workers = []
    for i in range(num_workers):
        worker = process_creation_func(
            target=chunked_worker_run, args=(map_func, args[i::num_workers], queues[i]), daemon=True
        )
        workers.append(worker)
        worker.start()

    for i in range(num_jobs):
        yield queues[i % num_workers].get()

    for worker in workers:
        worker.join()
        worker.close()

# #### this is the old version of chunked_multiprocess_run
def chunked_worker_old(worker_id, map_func, args, results_queue=None, init_ctx_func=None):
    ctx = init_ctx_func(worker_id) if init_ctx_func is not None else None
    for job_idx, arg in args:
        try:
            if not isinstance(arg, tuple) and not isinstance(arg, list):
                arg = [arg]
            if ctx is not None:
                res = map_func(*arg, ctx=ctx)
            else:
                res = map_func(*arg)
            results_queue.put((job_idx, res))
        except:
            traceback.print_exc()
            results_queue.put((job_idx, None))

def chunked_multiprocess_run(
        map_func, args, num_workers=None, ordered=True,
        init_ctx_func=None, q_max_size=1000, multithread=False):
    if multithread:
        from multiprocessing.dummy import Queue, Process
    else:
        from multiprocessing import Queue, Process
    args = zip(range(len(args)), args)
    args = list(args)
    n_jobs = len(args)
    if num_workers is None:
        num_workers = int(os.getenv('N_PROC', os.cpu_count()))
    results_queues = []
    if ordered:
        for i in range(num_workers):
            results_queues.append(Queue(maxsize=q_max_size // num_workers))
    else:
        results_queue = Queue(maxsize=q_max_size)
        for i in range(num_workers):
            results_queues.append(results_queue)
    workers = []
    for i in range(num_workers):
        args_worker = args[i::num_workers]
        p = Process(target=chunked_worker_old, args=(
            i, map_func, args_worker, results_queues[i], init_ctx_func), daemon=True)
        workers.append(p)
        p.start()
    for n_finished in range(n_jobs):
        results_queue = results_queues[n_finished % num_workers]
        job_idx, res = results_queue.get()
        assert job_idx == n_finished or not ordered, (job_idx, n_finished)
        yield res
    for w in workers:
        w.join()
