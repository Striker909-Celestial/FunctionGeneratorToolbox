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
    def __init__(self, index: int, post_processor: Callable[..., _T], post_processor_args: dict, input_sources: list[int]):
        self.index = index
        self.post_processor = post_processor
        self.post_processor_args = post_processor_args
        self.input_sources = input_sources
        self.input_buffer = multiprocessing.Array(ctypes.py_object, [None for _ in input_sources])
        self.request_sent = False

        self.total_time = multiprocessing.Value('d', 0.0)
        self.num_calls = multiprocessing.Value('i', 0)

    def size(self):
        """
        The number of slots in the buffer.
        :return: The number of slots in the buffer
        """
        return len(self.input_sources)

    def count(self):
        """
        The number of slots in the buffer that are filled.
        :return: The number of filled slots in the buffer
        """
        n = 0
        for i in range(len(self.input_sources)):
            if self.input_buffer[i] is not None:
                n += 1
        return n

    def process_buffer(self) -> bool | tuple[int, _T]:
        """
        Attempts to run the post-processor on the buffer's contents.
        :return: False if the buffer is not full, otherwise the post-processor's output
        """
        if any(self.input_buffer[i] is None for i in range(len(self.input_sources))):
            # Returns false if the buffer is not full
            return False
        self.num_calls.value += 1
        inputs = []
        self.request_sent = False
        for i in range(len(self.input_sources)):
            inputs.append(self.input_buffer[i])
            self.input_buffer[i] = None
        output = self.post_processor(*inputs, **self.post_processor_args)
        return self.index, output

    def send_request(self, requests: dict[int, multiprocessing.Queue[int]]) -> bool:
        """
        Sends a request for all missing inputs for the buffer
        if a request has not already been made since the last time the buffer emptied.
        :param requests:
        :return:
        """
        if self.request_sent:
            return False
        for i, s in enumerate(self.input_sources):
            if self.input_buffer[i] is not None:
                continue
            requests[s].put(self.index)
        self.request_sent = True
        return True

    def add_token(self, token: tuple[int, ...], requests: dict[int, multiprocessing.Queue[int]],
                  auto_run_post_processor=True, max_wait=1.0, auto_request=True) -> bool | tuple[int, _T] | None:
        if token[1] is None:
            # Returns false if the token's value is none
            return False
        token_index = token[0]
        start = datetime.datetime.now()
        # Attempts to place the token in the buffer
        for i, s in enumerate(self.input_sources):
            if token_index == s and self.input_buffer[i] is None:
                self.input_buffer[i] = token[1]
                break
            if i == len(self.input_sources) - 1:
                # Returns false if the token cannot be placed in the buffer
                self.total_time.value += (datetime.datetime.now() - start).total_seconds()
                return False
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

def buffered_queue_manager[_T](target_num: int, output_index: int, queue: multiprocessing.Queue,
                        buffers: multiprocessing.Array[Buffer], requests: dict[int, multiprocessing.Queue[int]],
                        output: multiprocessing.Queue, buffer_probabilities: dict[int, list[float]],
                        generator: Callable[..., _T], generator_args: dict,
                        num_generator_calls: multiprocessing.Value[int], total_generator_time: multiprocessing.Value[float],
                        num_loops: multiprocessing.Value[int], max_wait=1.0):
    while output.qsize() < target_num:
        num_loops.value += 1
        try:
            # briefly sleeps every loop if the processes are nearing completion to ward against overproduction
            if output.qsize() / target_num > 0.9:
                time.sleep(0.01)
            # Checks if a new token needs to be generated
            generator_requested = not requests[-1].empty()
            if generator_requested or queue.qsize() == 0:
                num_generator_calls.value += 1
                g_start = datetime.datetime.now()
                generated_token = (-1, func_timeout(max_wait, queued_generator, args=(generator, generator_args)))
                if generated_token[1] is None:
                    continue
                if generator_requested:
                    # Sends the generated token to the proper buffer it is requested
                    request = requests[-1].get()
                    result = buffers[request].add_token(generated_token, requests, max_wait=max_wait)
                    if result is False:
                        queue.put(generated_token)
                    elif result is not True and result is not None:
                        queue.put(result)
                else:
                    # Sends the generated token to the queue if there are no requests
                    queue.put(generated_token)
                total_generator_time.value += (datetime.datetime.now() - g_start).total_seconds()
                continue

            # If a new token was not generated, fetches one from the queue
            token = queue.get()
            if token[1] is None:
                continue
            t_index = token[0]
            if t_index == output_index:
                # If the token's index matches the output index, sends it to the output queue
                output.put(token[1])
                continue
            if not requests[t_index].empty():
                # If there is a request for a token with the same index, sends it to the proper buffer
                request = requests[t_index].get()
                result = buffers[request].add_token(token, requests, max_wait=max_wait)
                if result is False:
                    queue.put(token)
                elif result is not True and result is not None:
                    queue.put(result)
                continue

            # If the token is not requested, selects a buffer to place it in at random
            p = buffer_probabilities[t_index]
            if 1.0 in p:
                b_index = p.index(1.0)
            else:
                b_index = int(np.random.choice(len(p), p=p))
            result = buffers[b_index].add_token(token, requests, max_wait=max_wait)
            if result is False:
                queue.put(token)
            elif result is not True and result is not None:
                queue.put(result)

        except Exception as e:
            print(f"Error in queue manager: {e}", flush=True)
            traceback.print_exc()

def buffered_parallel_processing[_T, _R](target_num: int,
                                generator: Callable[..., _T], generator_args: dict,
                                post_processors: list[Callable], post_processors_args: list[dict],
                                post_processors_input_sources: list[list[int]],
                                num_processes=multiprocessing.cpu_count(), json_path: str | None = None,
                                max_wait=1.0, desc="Processing",
                                debug=False, debug_buffers=False, debug_requests=False, debug_frequency=0.5):
    print(f"Setting up {desc}...", end="")
    start = datetime.datetime.now()
    num_generator_calls = multiprocessing.Value('i', 0)
    total_generator_time = multiprocessing.Value('d', 0.0)
    num_loops = multiprocessing.Value('i', 0)

    queue = multiprocessing.Queue()
    num_post_processors = len(post_processors)
    output_index = num_post_processors - 1
    buffers = [Buffer(i, post_processors[i], post_processors_args[i], post_processors_input_sources[i]) for i in range(num_post_processors)]
    requests: dict[int, multiprocessing.Queue[int]] = {i: multiprocessing.Queue() for i in range(-1, num_post_processors)}
    output = multiprocessing.Queue()

    buffer_probabilities = {i: [float(post_processors_input_sources[i].count(j)) for j in range(num_post_processors)] for i in range(-1, num_post_processors)}
    for i, p in buffer_probabilities.items():
        s = len(post_processors_input_sources[i])
        p = [p_ / s for p_ in p]
        buffer_probabilities[i] = p
    for i, input_sources in enumerate(post_processors_input_sources):
        for j in input_sources:
            buffer_probabilities[j][i] += 1.0
    for p in buffer_probabilities.values():
        s = sum(p)
        if s == 0.0:
            continue
        for i in range(len(p)):
            p[i] /= s

    processors: list[multiprocessing.Process] = []
    for _ in tqdm(range(num_processes), desc="Starting Processes", bar_format="\x1B[38;5;115m{l_bar}\x1B[38;5;121m{bar}\x1B[38;5;115m{r_bar}\x1b[39m"):
        p = multiprocessing.Process(target=buffered_queue_manager, args=(
                        target_num, output_index, queue,
                        buffers, requests, output, buffer_probabilities,
                        generator, generator_args, num_generator_calls, total_generator_time,
                        num_loops, max_wait))
        p.start()
        processors.append(p)
    t = datetime.datetime.now()
    p_start = datetime.datetime.now()
    debug_n = 0
    with (tqdm(total=target_num, desc=desc, position=debug_n, bar_format="\x1B[38;5;115m{l_bar}\x1B[38;5;121m{bar}\x1B[38;5;115m{r_bar}\x1b[39m") as pbar):
        debug_n += 1
        if debug:
            loops_debug = tqdm(total=1000, desc="0.0", position=debug_n, bar_format="\x1B[38;5;73m  [Debug]\x1B[38;5;245m loops: \x1B[38;5;250m{desc}\x1B[38;5;245mloops/s\x1b[39m", leave=False)
            debug_n += 1
            generator_debug = tqdm(total=1000, desc="0.0", position=debug_n, bar_format="\x1B[38;5;73m  [Debug]\x1B[38;5;245m " + generator.__name__ + ": \x1B[38;5;250m{desc}\x1B[38;5;245mit/s\x1b[39m", leave=False)
            debug_n += 1
            post_processors_debug = [
                tqdm(total=1000, desc="0.0", position=debug_n + i,
                     bar_format="\x1B[38;5;73m  [Debug]\x1B[38;5;245m " + post_processors[i].__name__ + ": \x1B[38;5;250m{desc}\x1B[38;5;245mit/s\x1b[39m", leave=False)
                for i in range(num_post_processors)
            ]
            debug_n += len(post_processors_debug)
        if debug_buffers:
            buffers_debug = [
                tqdm(total=len(post_processors_input_sources[i]), desc=f"\x1B[38;5;98m  [Buffer]\x1B[38;5;245m {post_processors[i].__name__}",
                     position=debug_n + i,
                     bar_format="{l_bar}\x1B[38;5;250m{bar}\x1B[38;5;245m{n_fmt}/{total_fmt}\x1b[39m", leave=False)
                for i in range(num_post_processors)
            ]
            debug_n += len(buffers_debug)
        if debug_requests:
            requests_debug = [tqdm(total=1000, desc="0", position=debug_n,
                     bar_format="\x1B[38;5;90m  [Requests]\x1B[38;5;245m " + generator.__name__ + ": \x1B[38;5;250m{desc}\x1B[38;5;245mreqs\x1b[39m", leave=False)]
            requests_debug.extend([
                tqdm(total=1000, desc="0", position=debug_n + 1 + i,
                     bar_format="\x1B[38;5;90m  [Requests]\x1B[38;5;245m " + post_processors[i].__name__ + ": \x1B[38;5;250m{desc}\x1B[38;5;245mreqs\x1b[39m",
                     leave=False)
                for i in range(num_post_processors)
            ])
            debug_n += 1 + len(buffers_debug)
        while output.qsize() < target_num:
            #print(queue.qsize())
            ct = datetime.datetime.now()
            if (ct - t).total_seconds() > debug_frequency:
                t = ct
                pbar.n = output.qsize()
                pbar.refresh()
                if debug:
                    l_it_s = num_loops.value / (ct - p_start).total_seconds() if (ct - p_start).total_seconds() > 0.0 else -0.0
                    loops_debug.set_description_str(f"{l_it_s:.2f}")
                    g_it_s = num_generator_calls.value / total_generator_time.value if total_generator_time.value > 0.0 else -0.0
                    generator_debug.set_description_str(f"{g_it_s:.2f}")
                    for i in range(num_post_processors):
                        pp_it_s = buffers[i].get_process_rate()
                        post_processors_debug[i].set_description_str(f"{pp_it_s:.2f}")
                if debug_buffers:
                    for i in range(num_post_processors):
                        buffers_debug[i].n = buffers[i].count()
                        buffers_debug[i].refresh()
                if debug_requests:
                    requests_debug[0].n = requests[-1].qsize()
                    requests_debug[0].refresh()
                    for i in range(num_post_processors):
                        requests_debug[i + 1].n = requests[i].qsize()
                        requests_debug[i + 1].refresh()
        pbar.n = output.qsize()
        pbar.refresh()
    p_elapsed = (datetime.datetime.now() - p_start).total_seconds()
    if debug:
        print(f"\x1B[38;5;73m  [Debug]\x1B[38;5;245m loops: \x1B[38;5;250m{num_loops.value / p_elapsed:.2f}\x1B[38;5;245mit/s\x1b[39m")
        print(f"\x1B[38;5;73m  [Debug]\x1B[38;5;245m {generator.__name__}: \x1B[38;5;250m{num_generator_calls.value / total_generator_time.value:.2f}\x1B[38;5;245mit/s\x1b[39m")
        for i, post_processor in enumerate(post_processors):
            print(f"\x1B[38;5;73m  [Debug]\x1B[38;5;245m {post_processor.__name__}: \x1B[38;5;250m{buffers[i].get_process_rate():.2f}\x1B[38;5;245mit/s\x1b[39m")
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