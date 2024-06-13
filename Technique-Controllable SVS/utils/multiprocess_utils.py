import os
import traceback
from functools import partial
from tqdm import tqdm


def chunked_worker_binarize(worker_id, map_func, args, results_queue=None, init_ctx_func=None):
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
    def __init__(self, num_workers=None, init_ctx_func=None, multithread=False):
        if multithread:
            from multiprocessing.dummy import Queue, Process
        else:
            from multiprocessing import Queue, Process
        if num_workers is None:
            num_workers = int(os.getenv('N_PROC', os.cpu_count()))
        self.num_workers = num_workers
        self.results_queue = Queue(maxsize=-1)
        self.args_queue = Queue(maxsize=-1)
        self.workers = []
        self.total_jobs = 0
        for i in range(num_workers):
            p = Process(target=chunked_worker,
                        args=(i, self.args_queue, self.results_queue, init_ctx_func),
                        daemon=True)
            self.workers.append(p)
            p.start()

    def add_job(self, func, args):
        self.args_queue.put((self.total_jobs, func, args))
        self.total_jobs += 1

    def get_results(self):
        for w in range(self.num_workers):
            self.args_queue.put("<KILL>")
        self.n_finished = 0
        while self.n_finished < self.total_jobs:
            job_id, res = self.results_queue.get()
            yield job_id, res
            self.n_finished += 1
        for w in self.workers:
            w.join()

    def __len__(self):
        return self.total_jobs


def multiprocess_run_tqdm(map_func, args, num_workers=None, ordered=True, init_ctx_func=None,
                          multithread=False, desc=None):
    for i, res in tqdm(enumerate(
            multiprocess_run(map_func, args, num_workers, ordered, init_ctx_func, multithread)),
            total=len(args), desc=desc):
        yield i, res


def multiprocess_run(map_func, args, num_workers=None, ordered=True, init_ctx_func=None, multithread=False):
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
    manager = MultiprocessManager(num_workers, init_ctx_func, multithread)
    for arg in args:
        manager.add_job(map_func, arg)
    if ordered:
        n_jobs = len(args)
        results = ['<WAIT>' for _ in range(n_jobs)]
        i_now = 0
        for job_i, res in manager.get_results():
            results[job_i] = res
            while i_now < n_jobs and (not isinstance(results[i_now], str) or results[i_now] != '<WAIT>'):
                yield results[i_now]
                i_now += 1
    else:
        for res in manager.get_results():
            yield res



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
        p = Process(target=chunked_worker_binarize, args=(
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

