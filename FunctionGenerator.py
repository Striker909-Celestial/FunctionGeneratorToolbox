import datetime
import multiprocessing
from multiprocessing import freeze_support
from sympy import *
import numpy as np
import json

from pebble import ProcessPool
from tqdm import tqdm
import functools

functions = []
weights = []

def load_functions_dict(path: str):
    """
    Loads the function list with functions from a JSON file.
    
    Assumes the JSON file consists of a list of dicts, 
    each with the following keys:
    
    **func** - A SymPy expression representing the function
    
    **num_params** - The number of different free symbols in the function
    
    **weight**- The weight for the function in the random selection
    
    :param path: The path to the JSON file containing the function list
    """
    global functions, weights
    with open(path, 'r') as f:
        functions = json.load(f)
        for fn in functions:
            fn["func"] = sympify(fn["func"])
    weights = np.array([fn["weight"] for fn in functions])
    weights = weights / weights.sum()

def load_functions_list(path: str):
    """
    Loads the function list with functions from a JSON file.
    
    Assumes the JSON file consists of a list of strings, 
    each a SymPy expression that will be treated as **func**. 
    **num_params** is found by counting the number of free symbols in the expression.
    All **weight**s are set to 1.0.
    
    **func** - A SymPy expression representing the function
    
    **num_params** - The number of different free symbols in the function
    
    **weight**- The weight for the function in the random selection
    
    :param path: The path to the JSON file containing the function list
    :return: 
    """
    global functions
    global weights
    with open(path, 'r') as f:
        functions_list = json.load(f)
        for func in functions_list:
            functions.append(
                {
                    "func": func,
                    "num_params": len(func.free_symbols),
                    "weight": 1.0,
                }
            )
    w = 1.0 / len(functions)
    weights = np.array([w for _ in functions])

def random_function(weighted=True):
    """
    Returns a random function from the function list.
    :param weighted: If the random selection should be weighted or not
    :return: A function at random
    """
    if weighted:
        return functions[np.random.choice(len(functions), p=weights)]
    return functions[np.random.randint(len(functions))]

def standard_constant_chance(d):
    return (10.0 - d)/20.0
def standard_single_variable_chance(d):
    return (10.0 - d) / 20.0

def construct_random_scalar_function(params: int | list, weighted=True, max_depth=5,
                                      constant_chance=standard_constant_chance, constant_range=(-10,10),
                                      single_variable_chance=standard_single_variable_chance, complex_functions=False):
     """
     Constructs a random scalar function from recursively applying functions from the function list.
     
     :param params: A number of parameters for the function to have or a list of symbols to use as parameters
     :param weighted: If the random selection of functions used to construct the final function should be weighted
     :param max_depth: The maximum depth of the function to be constructed (i.e., a*(b*(c*(d*e))) has depth 4)
     :param constant_chance: The chance of a constant being used as an input, as a function of the maximum depth - the current depth
     :param constant_range: The range of constants to be used
     :param single_variable_chance: The chance of a single variable being used as an input, as a function of the maximum depth - the current depth
     :param complex_functions: If the constructed functions can be complex, or if only the real portion of it should be returned
     :return: A randomly constructed scalar function
     """
     if type(params) is int:
         params = symbols('x:'+str(params), real=not complex_functions)
     func = random_function(weighted)
     inputs = []
     for i in range(func["num_params"]):
         if np.random.rand() < single_variable_chance(max_depth):
             inputs.append(params[np.random.randint(len(params))])
         elif max_depth == 0 or np.random.rand() < constant_chance(max_depth):
             inputs.append(round(np.random.uniform(*constant_range), 2))
         else:
             inputs.append(construct_random_scalar_function(params, weighted, max_depth-1, constant_chance, constant_range, single_variable_chance))
     f_new = func['func'].subs([(symb, inputs[i]) for i, symb in enumerate(func['func'].free_symbols)])
     if not complex_functions:
         return re(f_new)
     return f_new

def construct_random_scalar_function_set(num_functions: int, params: int, weighted=True, max_depth=10,
                                      constant_chance=standard_constant_chance, constant_range=(-10,10),
                                      single_variable_chance=standard_single_variable_chance, complex_functions=False,
                                      overshoot=0.75, num_processes: int | None = None, chunk_size: int = 10000, max_wait=1.0) -> set:
    """
        Constructs a set of random scalar functions from recursively applying functions from the function list.

        This method parallelizes **construct_random_scalar_function**.

        As this method eliminates duplicate functions, the number of returned functions may be less than *num_functions*.
        For this reason, the **overshoot** parameter will increase the number of functions returned to attempt to overshoot 
        the desired number of functions.
        
        Will print telemetry readouts to the terminal during processing.

        :param num_functions: The number of functions to be constructed
        :param params: A number of parameters for the function to have
        :param weighted: If the random selection of functions used to construct the final functions should be weighted
        :param max_depth: The maximum depth of the functions to be constructed (i.e., a*(b*(c*(d*e))) has depth 4)
        :param constant_chance: The chance of a constant being used as an input, as a function of the maximum depth - the current depth
        :param constant_range: The range of constants to be used
        :param single_variable_chance: The chance of a single variable being used as an input, as a function of the maximum depth - the current depth
        :param complex_functions: If the constructed functions can be complex, or if only the real portion of them should be returned
        :param overshoot: A multiple of the desired number of functions to attempt to overshoot by: true_num_functions = num_functions * (1 + overshoot)
        :param num_processes: The maximum number of processes to use for parallelization, if None, will use all available processes
        :param chunk_size: The number of functions to be constructed in each chunk (<=10000 is suggested to avoid freezing)
        :param max_wait: The maximum time to wait for a function to be constructed before moving on to the next one
        :return: A set of random scalar functions
        """
    construct_fn_partial = functools.partial(construct_random_scalar_function, weighted=weighted, max_depth=max_depth,
                                              constant_chance=constant_chance, constant_range=constant_range,
                                              single_variable_chance=single_variable_chance,
                                              complex_functions=complex_functions)
    output = set()
    num_chunks = int(num_functions * (1.0 + overshoot) / chunk_size) + 1
    functions_per_chunk = int(num_functions * (1.0 + overshoot) / num_chunks)
    print(f"Constructing {functions_per_chunk * num_chunks} functions in {num_chunks} chunks")
    inputs = [params for i in range(functions_per_chunk)]
    start_time = datetime.datetime.now()
    for i in range(num_chunks):
        results = []
        print(f"Constructing  a chunk of {functions_per_chunk} functions: {i + 1} / {num_chunks}")
        try:
            with ProcessPool(max_workers=num_processes if num_processes is not None else multiprocessing.cpu_count(),
                             max_tasks=1000) as pool:
                future = pool.map(construct_fn_partial, inputs, timeout=max_wait)
                iterator = future.result()
                for _ in tqdm(inputs, desc="Constructing Functions"):
                    try:
                        results.append(next(iterator))
                    except:
                        pass
        except:
            continue
        output |= set(results)
    end_time = datetime.datetime.now()
    print(f"Constructed {len(output)} functions.\n"
          f"Time elapsed: {(datetime.datetime(1, 1, 1) + (end_time - start_time)).strftime("%H:%M:%S")}")
    return output

def test_constant(function: Expr, num_test_points=4, tolerance=0.001) -> bool:
    """
    Tests if a given function evaluates to a constant by testing it over several points.
    
    If the function returns the same value at all test points, it is considered constant.
    
    :param function: The function to be tested
    :param num_test_points: The number of test points to use
    :param tolerance: The maximum percent difference between two values to be considered equal (i.e., a tolerance of 0.001 will consider difference of less than 0.1% to be equal)
    :return: If the tested function is constant
    """
    results = []
    test_points = list(np.random.uniform(-100, 100, num_test_points))
    for p in test_points:
        results.append(function.subs(symbols('a'), p).evalf())
    avg = sum(results) / float(len(results))
    return all(abs((result - avg) / avg) < tolerance for result in results)

def prune_function(function: Expr, prune_constants=True, num_test_points=4, tolerance=0.001, complex_functions=False, round_n=2) -> Expr | None:
    """
    Simplifies and tests a function, using certain criteria to determine if it should be discarded.
    
    If a function is discarded, None will be returned. 
    Otherwise, a simplified and rounded version of the function will be returned.
    
    :param function: The function to be simplified and tested
    :param prune_constants: If the function should be discarded if it evaluates to be constant by **test_constant**
    :param num_test_points: The number of test points to be used by **test_constant**
    :param tolerance: The tolerance to be used by **test_constant**
    :param complex_functions: If the function can be complex, or if only the real portion of it should be returned
    :param round_n: A number of decimal places to round the function to, if None, no rounding will be performed
    :return: None, or a simplified and rounded version of the function
    """
    try:
        func_copy = function.copy()
        a = symbols('a')
        func_copy = func_copy.subs([(symb, a) for symb in func_copy.free_symbols])
        if prune_constants and test_constant(func_copy, num_test_points, tolerance):
            return None
        func_new = simplify(function)
        if func_new == nan:
            return None
        if round_n is not None:
            func_new = func_new.xreplace({n: round(n, round_n) for n in func_new.atoms(Number)})
        if not complex_functions:
            return re(func_new)
        return func_new
    except:
        return None

def prune_function_list(function_list: list | set, prune_constants=True, num_test_points=4, tolerance=0.001, complex_functions=False, round_n: int | None = 2, num_processes: int | None = None, max_wait=1.0) -> set:
    """
        Simplifies and tests each function in a list, using certain criteria to determine if it should be discarded.
        
        This method parallelizes **prune_function**.
        
        Will print telemetry readouts to the terminal during processing.
        
        :param function_list: A list or set of functions to be simplified and tested
        :param prune_constants: If a function should be discarded if it evaluates to be constant by **test_constant**
        :param num_test_points: The number of test points to be used by **test_constant**
        :param tolerance: The tolerance to be used by **test_constant**
        :param complex_functions: If functions can be complex, or if only the real portion of them should be included
        :param round_n: A number of decimal places to round functions to, if None, no rounding will be performed
        :param num_processes: The maximum number of processes to use for parallelization, if None, will use all available processes
        :param max_wait: The maximum time to wait for a function to be simplified and tested before moving on to the next one
        :return: A set of pruned functions
        """
    prune_fn_partial = functools.partial(prune_function, prune_constants=prune_constants, num_test_points=num_test_points, tolerance=tolerance,
                                         complex_functions=complex_functions, round_n=round_n)
    results = []
    print(f"Pruning {len(function_list)} functions")
    start_time = datetime.datetime.now()
    with ProcessPool(max_workers=num_processes if num_processes is not None else multiprocessing.cpu_count()) as pool:
        future = pool.map(prune_fn_partial, function_list, timeout=max_wait)
        iterator = future.result()
        for _ in tqdm(function_list, desc="Pruning"):
            try:
                results.append(next(iterator))
            except TimeoutError:
                pass
    new_function_list = set(results)
    new_function_list.discard(None)
    end_time = datetime.datetime.now()
    print(f"Pruned to {len(new_function_list)} functions.\n"
          f"Time elapsed: {(datetime.datetime(1, 1, 1) + (end_time - start_time)).strftime("%H:%M:%S")}")

    return new_function_list

def generate_dataset(num_functions: int, directory_path: str,
                     params: int, overshoot: float = 0.75, weighted=True, max_depth=10,
                     constant_chance=standard_constant_chance, constant_range=(-10,10),
                     single_variable_chance=standard_single_variable_chance, complex_functions=False, num_test_points=4, tolerance=0.001,
                     prune_constants=True, round_n: None | int = 3, num_processes: int | None = None, max_generate_wait=1.0, max_prune_wait=1.0):
    """
    Generates a dataset of random scalar functions of a certain size using **construct_random_scalar_function_set**
    and **prune_function_list**. This dataset is saved to a JSON file as a list of strings.

    Two or more passes will be taken to fully generate the dataset.
    The first pass will generate the majority of the dataset but will intentionally undershoot the desired number of functions.
    The second pass will overshoot the remaining number of functions such that the dataset will be full after pruning.
    If the second pass does not fill the dataset, additional similar passes will be taken until the dataset is full.
    Any extraneous functions will be discarded.

    This method uses parallel processing to speed up the generation process.

    Will print telemetry readouts to the terminal during processing.

    :param num_functions: The number of functions to be generated to make up the dataset
    :param directory_path: The path to the directory where the dataset will be saved
    :param params: The number of parameters for the functions to have
    :param overshoot: A multiple of the desired number of functions to attempt to overshoot by: true_num_functions = num_functions * (1 + overshoot). This parameter is doubled in the second pass and onwards
    :param weighted: If the random selection of functions used to construct the final functions should be weighted
    :param max_depth: The maximum depth of the functions to be constructed (i.e., a*(b*(c*(d*e))) has depth 4)
    :param constant_chance: The chance of a constant being used as an input, as a function of the maximum depth - the current depth
    :param constant_range: The range of constants to be used
    :param single_variable_chance: The chance of a single variable being used as an input, as a function of the maximum depth - the current depth
    :param complex_functions: If the constructed functions can be complex, or if only the real portion of them should be returned
    :param prune_constants: If a function should be discarded if it evaluates to be constant by **test_constant**
    :param num_test_points: The number of test points to be used by **test_constant**
    :param tolerance: The tolerance to be used by **test_constant**
    :param round_n: A number of decimal places to round functions to, if None, no rounding will be performed
    :param num_processes: The maximum number of processes to use for parallelization, if None, will use all available processes
    :param max_generate_wait: The maximum time to wait for a function to be constructed before moving on to the next one
    :param max_prune_wait: The maximum time to wait for a pruned before moving on to the next one
    :return:
    """
    start_time = datetime.datetime.now()
    dataset = construct_random_scalar_function_set(num_functions, params, weighted, max_depth, constant_chance, constant_range, single_variable_chance, complex_functions, num_processes, overshoot, max_wait=max_generate_wait)
    dataset = prune_function_list(dataset, prune_constants=prune_constants, num_test_points=num_test_points, tolerance=tolerance, complex_functions=complex_functions, round_n=round_n, max_wait=max_prune_wait, num_processes=num_processes)
    while len(dataset) < num_functions:
        print(len(dataset))
        temp_dataset = construct_random_scalar_function_set(num_functions - len(dataset), params, weighted, max_depth, constant_chance, constant_range, single_variable_chance, complex_functions, num_processes, 2.5 * overshoot, max_wait=max_generate_wait)
        dataset |= prune_function_list(temp_dataset, prune_constants=prune_constants, complex_functions=complex_functions, round_n=round_n, max_wait=max_prune_wait, num_processes=num_processes)
    dataset = list(dataset)
    np.random.shuffle(dataset)
    dataset = dataset[:num_functions]
    print("\rDataset fully generated, saving to file")
    with open(f"{directory_path}/n{num_functions}-x{params}-y1.json", 'w') as f:
        json.dump([str(datum) for datum in dataset], f, indent=4)
    print(f"Dataset saved to {directory_path}/n{num_functions}-x{params}-y1.json")
    end_time = datetime.datetime.now()
    time_elapsed = (end_time - start_time)
    print(f"\nTotal Generation Time: {(datetime.datetime(1, 1, 1) + time_elapsed).strftime("%H:%M:%S")}"
          f"\nFunctions Generated per Second: {num_functions / time_elapsed.total_seconds():.2f}\n")
    return dataset

def main():
    load_functions_list("datasets/n1000-x1-y1.json")
    generate_dataset(
        100000,
        "datasets",
        1,
        weighted=True,
        max_depth=5,
    )

if __name__ == "__main__":
    freeze_support()
    main()