import ctypes
import json
import time
import traceback
from typing import Callable
from tqdm import tqdm
import multiprocessing
from func_timeout import func_timeout, FunctionTimedOut
import datetime
import numpy as np
from collections import Counter

from Processor import Processor

def queued_generator[_T](generator: Callable[..., _T], generator_args: dict) -> tuple[int, _T]:
    """
    Generates an object of type **_T** using the generator.
    Returns a token with the index 0 and the generated object.

    :param generator: The generator to generate the object with
    :param generator_args: The args for the generator
    :return: A token containing the generated object
    """
    output = generator(**generator_args)
    return 0, output

def queued_post_processor[_T](token, post_processors: list[Callable], post_processors_args: list[dict]) -> tuple[int, _T]:
    """
    Applies the post-processor at the token's index to the object in the token.
    Increments the index of the token by 1 before returning it.

    :param token: The token containing the object to be processed and an index
    :param post_processors: A list of post-processors to apply to the object
    :param post_processors_args: A list of args for the post-processors
    :return: A token containing the object with the correct post-processor applied
    """
    index = token[0]
    output = post_processors[index](token[1], **post_processors_args[index])
    return index + 1, output

def queue_manager_debug[_T](target_num: int, queue: multiprocessing.Queue, output: multiprocessing.Queue,
                      generator_total_time: multiprocessing.Value, num_generator_calls: multiprocessing.Value,
                      post_processor_total_time: multiprocessing.Array, num_post_processor_calls: multiprocessing.Array,
                      generator: Callable[..., _T], generator_args: dict,
                      post_processors: list[Callable], post_processors_args: list[dict], max_wait):
    """
    A single worker process that manages the queue and output.

    Coordinates generating new tokens and applying post-processors to them until the output queue is full.

    :param target_num: The target number of outputs to generate and process
    :param queue: The queue of all tokens to be processed
    :param output: The queue of all processed outputs
    :param generator_total_time: The total time spent generating tokens
    :param num_generator_calls: The number of times the generator has been called
    :param post_processor_total_time: An array of the total time spent by each post-processor
    :param num_post_processor_calls: An array of the number of times each post-processor has been called
    :param generator: The generator to generate new objects with
    :param generator_args: The args for the generator
    :param post_processors: A list of post-processors to apply to the objects
    :param post_processors_args: A list of args for the post-processors
    :param max_wait: The maximum time to wait for a generator or post-processor task before that task is cancelled
    """
    while output.qsize() < target_num:
        try:
            if output.qsize() / target_num > 0.9:
                time.sleep(0.01)
            if queue.qsize() == 0:
                start = datetime.datetime.now()
                generated_token = func_timeout(max_wait, queued_generator, args=(generator, generator_args))
                end = datetime.datetime.now()
                generator_total_time.value += int((end - start).total_seconds() * 1000)
                num_generator_calls.value += 1
                if generated_token[1] is None:
                    continue
                queue.put(generated_token)
                continue
            token = queue.get()
            index = token[0]
            start = datetime.datetime.now()
            new_token = func_timeout(max_wait, queued_post_processor, args=(token, post_processors, post_processors_args))
            end = datetime.datetime.now()
            post_processor_total_time[index] += int((end - start).total_seconds() * 1000)
            num_post_processor_calls[index] += 1
            if new_token[0] >= len(post_processors):
                output.put(new_token[1])
                continue
            if new_token[1] is None:
                continue
            queue.put(new_token)
        except FunctionTimedOut:
            pass

def queue_manager[_T](target_num: int, queue: multiprocessing.Queue, output: multiprocessing.Queue,
                      generator: Callable[..., _T], generator_args: dict,
                      post_processors: list[Callable], post_processors_args: list[dict], max_wait):
    """
    A single worker process that manages the queue and output.

    Coordinates generating new tokens and applying post-processors to them until the output queue is full.

    :param target_num: The target number of outputs to generate and process
    :param queue: The queue of all tokens to be processed
    :param output: The queue of all processed outputs
    :param generator: The generator to generate new objects with
    :param generator_args: The args for the generator
    :param post_processors: A list of post-processors to apply to the objects
    :param post_processors_args: A list of args for the post-processors
    :param max_wait: The maximum time to wait for a generator or post-processor task before that task is cancelled
    """
    while output.qsize() < target_num:
        try:
            if output.qsize() / target_num > 0.9:
                time.sleep(0.01)
            if queue.qsize() == 0:
                generated_token = func_timeout(max_wait, queued_generator, args=(generator, generator_args))
                if generated_token[1] is None:
                    continue
                queue.put(generated_token)
                continue
            token = queue.get()
            new_token = func_timeout(max_wait, queued_post_processor, args=(token, post_processors, post_processors_args))
            if new_token[0] >= len(post_processors):
                output.put(new_token[1])
                continue
            if new_token[1] is None:
                continue
            queue.put(new_token)
        except FunctionTimedOut:
            pass

def queued_parallel_processing[_T, _R](target_num: int,
                                generator: Callable[..., _T], generator_args: dict,
                                post_processors: list[Callable], post_processors_args: list[dict],
                                num_processes=multiprocessing.cpu_count(), json_path: str | None = None,
                                max_wait=1.0, desc="Processing", debug=False, debug_frequency=0.5) -> list[_R]:
    """
    Generates and processes a given number of outputs using a queue system and multiprocessing.

    Will print telemetry readouts to the terminal during processing.

    :param target_num: The target number of outputs to generate and process
    :param generator: The generator to generate new objects with
    :param generator_args: The args for the generator
    :param post_processors: A list of post-processors to apply to the objects
    :param post_processors_args: A list of args for the post-processors
    :param num_processes: The number of processes to use, defaults to all available processes
    :param json_path: A path to save the results to as a JSON file, if None, results will not be saved to file
    :param max_wait: The maximum time to wait for a generator or post-processor task before that task is cancelled
    :param desc: A description of the processing being done for telemetry
    :param debug: If debug information should be displayed during processing describing the operation speeds of the generator and each post-processor
    :param debug_frequency: The number of seconds between updates to debug information while processing
    :return: An array of processed outputs
    """
    start = datetime.datetime.now()
    queue, output = multiprocessing.Queue(), multiprocessing.Queue()
    generator_total_time = multiprocessing.Value('i', 1)
    num_generator_calls = multiprocessing.Value('i', 1)
    post_processor_total_time = multiprocessing.Array('i', [1 for _ in range(len(post_processors))])
    num_post_processor_calls = multiprocessing.Array('i', [1 for _ in range(len(post_processors))])
    processors: list[multiprocessing.Process] = []
    for _ in range(num_processes):
        if debug:
            p = multiprocessing.Process(target=queue_manager_debug, args=(target_num, queue, output,
                                                                generator_total_time, num_generator_calls,
                                                                post_processor_total_time, num_post_processor_calls
                                                                , generator, generator_args,
                                                                post_processors, post_processors_args, max_wait))
        else:
            p = multiprocessing.Process(target=queue_manager, args=(target_num, queue, output,
                                                                    generator, generator_args,
                                                                    post_processors, post_processors_args, max_wait))
        p.start()
        processors.append(p)
    t = datetime.datetime.now()
    with tqdm(total=target_num, desc=desc, position=0, bar_format="\x1B[38;5;115m{l_bar}\x1B[38;5;121m{bar}\x1B[38;5;115m{r_bar}\x1b[39m") as pbar:
        if debug:
            generator_debug = tqdm(total=1000, desc="0.0", position=1, bar_format="\x1B[38;5;73m  [Debug]\x1B[38;5;245m " + generator.__name__ + ": \x1B[38;5;250m{desc}\x1B[38;5;245mit/s\x1b[39m", leave=False)
            post_processors_debug = [
                tqdm(total=1000, desc="0.0", position=2+i,
                     bar_format="\x1B[38;5;73m  [Debug]\x1B[38;5;245m " + post_processors[i].__name__ + ": \x1B[38;5;250m{desc}\x1B[38;5;245mit/s\x1b[39m", leave=False)
                for i in range(len(post_processors))
            ]
        while output.qsize() < target_num:
            ct = datetime.datetime.now()
            if (ct - t).total_seconds() > debug_frequency:
                t = ct
                pbar.update(output.qsize() - pbar.n)
                if debug:
                    g_it_s = float(1000 * num_generator_calls.value) / float(generator_total_time.value)
                    generator_debug.set_description_str(f"{g_it_s:.2f}")
                    for i in range(len(post_processors)):
                        pp_it_s = float(1000 * num_post_processor_calls[i]) / float(post_processor_total_time[i])
                        post_processors_debug[i].set_description_str(f"{pp_it_s:.2f}")
        pbar.update(output.qsize() - pbar.n)
    if debug:
        print(f"\x1B[38;5;73m  [Debug]\x1B[38;5;245m {generator.__name__}: \x1B[38;5;250m{float(1000 * num_generator_calls.value) / float(generator_total_time.value):.2f}\x1B[38;5;245mit/s\x1b[39m")
        for i, post_processor in enumerate(post_processors):
            print(f"\x1B[38;5;73m  [Debug]\x1B[38;5;245m {post_processor.__name__}: \x1B[38;5;250m{float(1000 * num_post_processor_calls[i]) / float(post_processor_total_time[i]):.2f}\x1B[38;5;245mit/s\x1b[39m")
    results: list[_R] = []
    with tqdm(total=target_num, desc="Packaging Outputs", bar_format="\x1B[38;5;115m{l_bar}\x1B[38;5;121m{bar}\x1B[38;5;115m{r_bar}\x1b[39m") as pbar:
        while len(results) < target_num:
            results.append(output.get())
            pbar.update()
    with tqdm(total=num_processes, desc="Cleaning Up", bar_format="\x1B[38;5;115m{l_bar}\x1B[38;5;121m{bar}\x1B[38;5;115m{r_bar}\x1b[39m") as pbar:
        for p in processors:
            p.join(timeout=max_wait)
            p.terminate()
            pbar.update()
    if json_path is not None:
        print(f"\x1B[38;5;245mSaving results to \x1B[38;5;250m{json_path}\x1b[39m")
        json_results = []
        for result in results:
            json_results.append(str(result))
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=4)
    end = datetime.datetime.now()
    time_elapsed = (end - start)
    print(f"\n\x1B[38;5;245mTotal {desc} Time: \x1B[38;5;250m{(datetime.datetime(1, 1, 1) + time_elapsed).strftime("%H:%M:%S")}"
          f"\n\x1B[38;5;245mIterations per Second: \x1B[38;5;250m{target_num / time_elapsed.total_seconds():.2f}\x1b[39m")
    return results

# Buffer/graph-based parallel processing

class Buffer[_T]:
    def __init__(self, processor: Processor, n: int):
        self.processor = processor
        self.n = n
        self.name = self.processor.name + "~" + str(n)
        self.current_dependencies = self.processor.dependencies.copy()
        self.current_dependencies_map = {key: val.copy() for key, val in self.processor.dependencies_map.items()}

        self.input_buffer = {}

        self.request_sent = False

        self.total_time = multiprocessing.Value('d', 0.0)
        self.num_calls = multiprocessing.Value('i', 0)

    def size(self):
        """
        The number of slots in the buffer.
        :return: The number of slots in the buffer
        """
        return self.processor.num_dependant_args

    def count(self):
        """
        The number of slots in the buffer that are filled.
        :return: The number of filled slots in the buffer
        """
        return len(self.input_buffer)

    def process_buffer(self) -> bool | tuple[int, _T]:
        """
        Attempts to run the post-processor on the buffer's contents.
        :return: False if the buffer is not full, otherwise the post-processor's output
        """
        if self.count() < self.size():
            # Returns false if the buffer is not full
            return False
        self.num_calls.value += 1
        self.request_sent = False
        self.current_dependencies = self.processor.dependencies.copy()
        self.current_dependencies_map = {key: val.copy() for key, val in self.processor.dependencies_map.items()}
        output = self.processor.process(self.input_buffer)
        #print(self.name, output)
        self.input_buffer = {}
        return output

    def send_request(self, requests: dict[str, multiprocessing.Queue[str]]) -> bool:
        """
        Sends a request for all missing inputs for the buffer
        if a request has not already been made since the last time the buffer emptied.
        :param requests:
        :return:
        """
        if self.request_sent:
            return False
        requested_kws = set()
        for dep in self.current_dependencies:
            for kw in self.current_dependencies_map[dep]:
                if kw in requests:
                    continue
                requested_kws.add(kw)
                requests[dep].put(self.name)
        return True

    def add_token(self, token: tuple[str, ...], requests: dict[str, multiprocessing.Queue[str]],
                  auto_run_post_processor=True, max_wait=1.0, auto_request=True) -> bool | tuple[str, _T] | None:
        if token[1] is None:
            # Returns false if the token's value is none
            return False
        token_source = token[0]
        if token_source not in self.current_dependencies:
            # Returns the token if it is not needed
            return token
        start = datetime.datetime.now()
        kw = list(self.current_dependencies_map[token_source])[0]
        for dep in self.current_dependencies_map.keys():
            if len(self.current_dependencies_map[dep]) == 0:
                continue
            self.current_dependencies_map[dep].remove(kw)
            if len(self.current_dependencies_map[dep]) == 0:
                self.current_dependencies.remove(dep)
        # Places the token's data in the buffer
        self.input_buffer[kw] = token[1]

        if auto_run_post_processor:
            process_output = func_timeout(max_wait, self.process_buffer)
            if process_output is not False:
                self.total_time.value += (datetime.datetime.now() - start).total_seconds()
                return process_output
        if auto_request:
            self.send_request(requests)
        self.total_time.value += (datetime.datetime.now() - start).total_seconds()
        return True

    def get_process_rate(self) -> float:
        if self.total_time.value == 0.0:
            return -0.0
        return self.num_calls.value / self.total_time.value
    
def buffered_queue_generator_manager(queue: multiprocessing.Queue, active: multiprocessing.Value[bool],
                        buffers: dict[str, Buffer], requests: dict[str, multiprocessing.Queue[str]],
                        generators: list[Processor], generator_probabilities: list[float],
                        num_generator_calls: dict[str, float], generator_time: dict[str, float],
                        num_loops: multiprocessing.Value[int], max_wait=1.0, target_queue_size=10):
    while active.value:
        num_loops.value += 1
        try:
            # Fills the first request for each generator if that generator is requested
            for generator in generators:
                g_name = generator.name
                if requests[g_name].qsize() > 0:
                    request = requests[g_name].get()
                    num_generator_calls[g_name].value += 1
                    g_start = datetime.datetime.now()
                    generated_token = func_timeout(max_wait, generator.process)
                    generator_time[g_name].value += (datetime.datetime.now() - g_start).total_seconds()
                    if generated_token[1] is None:
                        continue
                    result = buffers[request].add_token(generated_token, requests, max_wait=max_wait)
                    if result is False:
                        queue.put(generated_token)
                    elif result is not True and result is not None:
                        queue.put(result)

            # Generates and adds a new token to the queue while the queue is under the target size
            p = [gp.value for gp in generator_probabilities]
            while queue.qsize() < target_queue_size:
                generator = np.random.choice(generators, p=p)
                g_name = generator.name
                num_generator_calls[g_name].value += 1
                g_start = datetime.datetime.now()
                generated_token = func_timeout(max_wait, generator.process)
                generator_time[g_name].value += (datetime.datetime.now() - g_start).total_seconds()
                if generated_token[1] is None:
                    break
                queue.put(generated_token)
        except TimeoutError:
            continue
        except Exception as e:
            print(f"Error in generator queue manager: {e}", flush=True)
            traceback.print_exc()
            

def buffered_queue_post_processor_manager(queue: multiprocessing.Queue, active: multiprocessing.Value[bool],
                        buffers: dict[str, Buffer], outputs: dict[str, multiprocessing.Queue],
                        buffer_output_probabilities: dict[str, dict[str, float]],
                        requests: dict[str, multiprocessing.Queue[str]],
                        num_loops: multiprocessing.Value[int], max_wait=1.0):
    while active.value:
        num_loops.value += 1
        try:
            # briefly sleeps every loop if the processes are nearing completion to ward against overproduction
            #if output.qsize() / target_num > 0.9:
            #    time.sleep(0.01)
            token = queue.get()
            t_source = token[0]
            if not requests[t_source].empty():
                # If there is a request for a token with the same source, sends it to the proper buffer
                request = requests[t_source].get()
                result = buffers[request].add_token(token, requests, max_wait=max_wait)
                if type(result) is not bool and result is not None:
                    queue.put(result)
                continue

            # If the token is not requested, selects a buffer or output to place it in at random
            p = buffer_output_probabilities[t_source]
            target = np.random.choice(list(p.keys()), p=[gp.value for gp in p.values()])
            if target in outputs.keys():
                outputs[target].put(token)
                continue
            result = buffers[str(target)].add_token(token, requests, max_wait=max_wait)
            if type(result) is not bool and result is not None:
                queue.put(result)

        except Exception as e:
            print(f"Error in queue manager: {e}", flush=True)
            traceback.print_exc()

def buffered_parallel_processing(output_targets: dict[str, int],
                                processors: list[Processor], processor_counts: dict[str, int] | None=None,
                                num_generator_processes: int | None=None,
                                num_post_processor_processes: int | None=None,
                                json_path: str | None = None,
                                max_wait=1.0, target_queue_size=10, desc="Processing",
                                debug_outputs=True, debug_queue=False, debug_rates=False, debug_buffers=False, debug_requests=False,
                                debug_frequency=0.5):
    print(f"Setting up {desc}...", end="")
    start = datetime.datetime.now()

    if processor_counts is None:
        processor_counts = {processor.name: 1 for processor in processors}

    generators = []
    post_processors = []
    buffers = {}
    for processor in processors:
        name = processor.name
        is_generator = processor.is_generator
        n = 0
        for i in range(processor_counts[name]):
            if is_generator:
                generators.append(processor)
            else:
                post_processors.append(processor)
                buffer = Buffer(processor, n)
                buffers[buffer.name] = buffer
                n += 1

    queue = multiprocessing.Queue()
    active = multiprocessing.Value('b', True)
    requests: dict[str, multiprocessing.Queue[str]] = {processor.name: multiprocessing.Queue() for processor in processors}
    outputs: dict[str, multiprocessing.Queue[str]] = {key + "~out": multiprocessing.Queue() for key in output_targets.keys()}
    total_target = sum(output_targets.values())

    generator_probabilities = [multiprocessing.Value('d', 1.0 / len(generators)) for _ in range(len(generators))]
    buffer_output_probabilities = {}
    for processor in processors:
        name = processor.name
        probabilities = {}
        if name in output_targets.keys():
            probabilities[name + "~out"] = 1.0
        for buffer in buffers.values():
            if name in buffer.current_dependencies:
                probabilities[buffer.name] = 1.0
        tot = float(len(probabilities))
        for key, prob in probabilities.items():
            probabilities[key] = multiprocessing.Value('d', prob / tot)
        buffer_output_probabilities[name] = probabilities

    num_generator_calls = {generator.name: multiprocessing.Value('i', 0) for generator in generators}
    generator_time = {generator.name: multiprocessing.Value('d', 0.0) for generator in generators}
    num_loops = multiprocessing.Value('i', 0)

    processes: list[multiprocessing.Process] = []
    num_threads = multiprocessing.cpu_count()
    if num_generator_processes is None and num_post_processor_processes is None:
        gen_ratio = float(len(generators)) / float(len(generators) + len(post_processors))
        num_generator_processes = int(gen_ratio * num_threads) + 1
    elif num_generator_processes is None:
        num_generator_processes = num_threads - num_post_processor_processes
    else:
        num_threads = num_generator_processes + num_post_processor_processes


    for i in tqdm(range(num_threads), desc="Starting Processes", bar_format="\x1B[38;5;115m{l_bar}\x1B[38;5;121m{bar}\x1B[38;5;115m{r_bar}\x1b[39m"):
        if i < num_generator_processes:
            proc = multiprocessing.Process(target=buffered_queue_generator_manager,
                                           kwargs={
                                               "queue": queue,
                                               "active": active,
                                               "buffers": buffers,
                                               "requests": requests,
                                               "generators": generators,
                                               "generator_probabilities": generator_probabilities,
                                               "num_generator_calls": num_generator_calls,
                                               "generator_time": generator_time,
                                               "num_loops": num_loops,
                                               "max_wait": max_wait,
                                             "target_queue_size": target_queue_size
                                           })
        else:
            proc = multiprocessing.Process(target=buffered_queue_post_processor_manager,
                                           kwargs={
                                               "queue": queue,
                                               "active": active,
                                               "buffers": buffers,
                                               "outputs": outputs,
                                               "buffer_output_probabilities": buffer_output_probabilities,
                                               "requests": requests,
                                               "num_loops": num_loops,
                                               "max_wait": max_wait
                                           })
        proc.start()
        processes.append(proc)
    t = datetime.datetime.now()
    p_start = datetime.datetime.now()
    
    debug_tqdms = {}
    debug_update_functions = {}

    def add_debug(key: str, tq: tqdm, func: Callable):
        debug_tqdms[key] = tq
        debug_update_functions[key] = func
    
    if debug_outputs:
        for key, target in output_targets.items():
            output_tq = tqdm(total=target, desc=key, position=len(debug_tqdms), bar_format="\x1B[38;5;115m{l_bar}\x1B[38;5;121m{bar}\x1B[38;5;115m{r_bar}\x1b[39m")
            def output_update_func():
                output_tq.n = outputs[key + "~out"].qsize()
                output_tq.refresh()

            add_debug(key + "~out", output_tq, output_update_func)

    if debug_queue:
        queue_tq = tqdm(total=target_queue_size * 10, desc="Queue Size", position=len(debug_tqdms), bar_format="\x1B[38;5;115m{l_bar}\x1B[38;5;121m{bar}\x1B[38;5;115m{r_bar}\x1b[39m")
        def queue_update_func():
            queue_tq.n = queue.qsize()
            output_tq.refresh()

        add_debug("queue", queue_tq, queue_update_func)
            
    if debug_rates:
        loops_tq = tqdm(total=1, desc="-0.0", position=len(debug_tqdms), bar_format="\x1B[38;5;73m  [Debug]\x1B[38;5;245m loops: \x1B[38;5;250m{desc}\x1B[38;5;245mloops/s\x1b[39m", leave=False)
        def loops_update_func():
            elapsed_time = (datetime.datetime.now() - p_start).total_seconds()
            if elapsed_time == 0.0:
                return
            loops_per_sec = float(num_loops.value) / elapsed_time
            loops_tq.set_description_str(f"{loops_per_sec:.2f}")

        add_debug("loops", loops_tq, loops_update_func)

        for generator in generators:
            gname = generator.name
            generator_rate_tq = tqdm(total=1, desc="-0.0", position=len(debug_tqdms),
                      bar_format="\x1B[38;5;73m  [Debug]\x1B[38;5;245m " + gname + ": \x1B[38;5;250m{desc}\x1B[38;5;245mit/s\x1b[39m", leave=False)
            def generator_rate_update_func():
                gtime = generator_time[gname].value
                if gtime == 0.0:
                    return
                rate = num_generator_calls[gname].value / gtime
                generator_rate_tq.set_description_str(f"{rate:.2f}")

            add_debug(name + "~rate", generator_rate_tq, generator_rate_update_func)
        for name, buffer in buffers.items():
            buffer_rate_tq = tqdm(total=1, desc="-0.0", position=len(debug_tqdms),
                      bar_format="\x1B[38;5;73m  [Debug]\x1B[38;5;245m " + name + ": \x1B[38;5;250m{desc}\x1B[38;5;245mit/s\x1b[39m",
                      leave=False)
            def buffer_rate_update_func():
                rate = buffer.get_process_rate()
                buffer_rate_tq.set_description_str(f"{rate:.2f}")

            add_debug(name + "~rate", buffer_rate_tq, buffer_rate_update_func)

    if debug_buffers:
        for name, buffer in buffers.items():
            buffer_tq = tqdm(total=buffer.size(), desc=f"\x1B[38;5;98m  [Buffer]\x1B[38;5;245m {name}",
                     position=len(debug_tqdms), bar_format="{l_bar}\x1B[38;5;250m{bar}\x1B[38;5;245m{n_fmt}/{total_fmt}\x1b[39m", leave=False)
            def buffer_update_func():
                buffer_tq.n = buffer.count()
                buffer_tq.refresh()

            add_debug(name, buffer_tq, buffer_update_func)

    if debug_requests:
        for key in requests.keys():
            request_tq = tqdm(total=1, desc="0", position=len(debug_tqdms),
                     bar_format="\x1B[38;5;90m  [Requests]\x1B[38;5;245m " + key + ": \x1B[38;5;250m{desc}\x1B[38;5;245mreqs\x1b[39m", leave=False)
            def requests_update_func():
                request_tq.set_description_str(str(requests[key].qsize()))

            add_debug(key + "~request", request_tq, requests_update_func)

    while any([outputs[key + "~out"].qsize() < target for key, target in output_targets.items()]):
        ct = datetime.datetime.now()
        if (ct - t).total_seconds() > debug_frequency:
            t = ct
            for debug in debug_tqdms.keys():
                debug_update_functions[debug]()
    p_elapsed = (datetime.datetime.now() - p_start).total_seconds()
    for debug in debug_tqdms.keys():
        debug_update_functions[debug]()

    results = {key: [] for key in output_targets.keys()}
    for key, target in tqdm(list(output_targets.items())[:total_target], desc="Packaging Outputs", bar_format="\x1B[38;5;115m{l_bar}\x1B[38;5;121m{bar}\x1B[38;5;115m{r_bar}\x1b[39m"):
        for i in range(target):
            results[key].append(outputs[key + "~out"].get())
    for proc in tqdm(processes, desc="Cleaning Up", bar_format="\x1B[38;5;115m{l_bar}\x1B[38;5;121m{bar}\x1B[38;5;115m{r_bar}\x1b[39m"):
        proc.join(timeout=max_wait)
        proc.terminate()
    if json_path is not None:
        print(f"\x1B[38;5;245mSaving results to \x1B[38;5;250m{json_path}\x1b[39m")
        json_results = []
        for result in results:
            json_results.append(str(result))
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=4)
    end = datetime.datetime.now()
    time_elapsed = (end - start)
    print(f"\n\x1B[38;5;245mTotal {desc} Time: \x1B[38;5;250m{(datetime.datetime(1, 1, 1) + time_elapsed).strftime("%H:%M:%S")}"
          f"\n\x1B[38;5;245mIterations per Second: \x1B[38;5;250m{total_target / time_elapsed.total_seconds():.2f}\x1b[39m")
    return results