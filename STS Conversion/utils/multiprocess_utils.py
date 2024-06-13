import os
import traceback

from tqdm import tqdm


def chunked_worker(worker_id, map_func, args, results_queue=None, init_ctx_func=None):
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
        p = Process(target=chunked_worker, args=(
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


def chunked_worker2(worker_id, args_queue=None, results_queue=None, init_ctx_func=None):
    ctx = init_ctx_func(worker_id) if init_ctx_func is not None else None
    while True:
        args = args_queue.get()
        if args == '<KILL>':
            return
        job_id, map_func, arg = args
        try:
            if ctx is not None:
                res = map_func(*arg, ctx=ctx)
            else:
                res = map_func(*arg)
            results_queue.put((job_id, res))
        except:
            traceback.print_exc()
            results_queue.put((job_id, None))


class MultiprocessManager:
    def __init__(self, num_workers=None, init_ctx_func=None):
        from multiprocessing import Queue, Process
        if num_workers is None:
            num_workers = int(os.getenv('N_PROC', os.cpu_count()))
        self.num_workers = num_workers
        self.results_queue = Queue(maxsize=-1)
        self.args_queue = Queue(maxsize=-1)
        self.workers = []
        self.total_jobs = 0
        for i in range(num_workers):
            p = Process(target=chunked_worker2,
                        args=(i, self.args_queue, self.results_queue, init_ctx_func),
                        daemon=True)
            self.workers.append(p)
            p.start()

    def add_job(self, func, arg):
        self.args_queue.put((self.total_jobs, func, arg))
        self.total_jobs += 1

    def get_results(self):
        for w in range(self.num_workers):
            self.args_queue.put("<KILL>")
        results = [None for _ in range(self.total_jobs)]
        self.n_finished = 0
        t = tqdm(desc='MultiprocessManager Process: ', total=self.total_jobs)
        while self.n_finished < self.total_jobs:
            t.update()
            job_id, res = self.results_queue.get()
            results[job_id] = res
            self.n_finished += 1
        for w in self.workers:
            w.join()
        return results
