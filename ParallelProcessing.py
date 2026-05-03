import json
import time
from typing import Callable
from tqdm import tqdm
import multiprocessing
from func_timeout import func_timeout, FunctionTimedOut
import datetime

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

def queue_manager[_T](target_num: int, queue: multiprocessing.Queue, output: multiprocessing.Queue,
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

def queued_parallel_processing[_T, _R](target_num: int,
                                generator: Callable[..., _T], generator_args: dict,
                                post_processors: list[Callable], post_processors_args: list[dict],
                                num_processes=multiprocessing.cpu_count(), json_path: str | None = None,
                                max_wait=1.0, desc="Processing", debug=False) -> list[_R]:
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
        p = multiprocessing.Process(target=queue_manager, args=(target_num, queue, output,
                                                                generator_total_time, num_generator_calls,
                                                                post_processor_total_time, num_post_processor_calls
                                                                , generator, generator_args,
                                                                post_processors, post_processors_args, max_wait))
        p.start()
        processors.append(p)
    with tqdm(total=target_num, desc=desc, position=0, bar_format="\x1B[38;5;115m{l_bar}\x1B[38;5;121m{bar}\x1B[38;5;115m{r_bar}\x1b[39m") as pbar:
        if debug:
            generator_debug = tqdm(total=1000, desc="0.0", position=1, bar_format="\x1B[38;5;73m  [Debug]\x1B[38;5;245m " + generator.__name__ + ": \x1B[38;5;250m{desc}\x1B[38;5;245mit/s\x1b[39m", leave=False)
            post_processors_debug = [
                tqdm(total=1000, desc="0.0", position=2+i,
                     bar_format="\x1B[38;5;73m  [Debug]\x1B[38;5;245m " + post_processors[i].__name__ + ": \x1B[38;5;250m{desc}\x1B[38;5;245mit/s\x1b[39m", leave=False)
                for i in range(len(post_processors))
            ]
        while output.qsize() < target_num:
            pbar.update(output.qsize() - pbar.n)
            if debug:
                g_it_s = float(1000 * num_generator_calls.value) / float(generator_total_time.value)
                generator_debug.set_description_str(f"{g_it_s:.2f}")
                for i in range(len(post_processors)):
                    pp_it_s = float(1000 * num_post_processor_calls[i]) / float(post_processor_total_time[i])
                    post_processors_debug[i].set_description_str(f"{pp_it_s:.2f}")
        pbar.update()
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