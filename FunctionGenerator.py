import datetime
import multiprocessing
import traceback
from multiprocessing import freeze_support
from sympy import *
import numpy as np
import json

from pebble import ProcessPool
from tqdm import tqdm
import functools
import re as regex

import ParallelProcessing as parallel_processing
functions = []
weights = []

def load_functions_dict(path: str):
    """
    Loads the function list with functions from a JSON file.
    
    Assumes the JSON file consists of a list of dicts, 
    each with the following keys:
    
    **func** - A SymPy expression representing the function
    
    **num_symbols** - The number of different free symbols in the function
    
    **weight**- The weight for the function in the random selection
    
    :param path: The path to the JSON file containing the function list
    """
    global functions, weights
    with open(path, 'r') as f:
        functions = json.load(f)
        for fn in functions:
            fn["func"] = sympify(fn["func"])
            fn["num_symbols"] = len(fn["func"].free_symbols)
    weights = np.array([fn["weight"] for fn in functions])
    weights = weights / weights.sum()

def load_functions_list(path: str):
    """
    Loads the function list with functions from a JSON file.
    
    Assumes the JSON file consists of a list of strings, 
    each a SymPy expression that will be treated as **func**. 
    **num_symbols** is found by counting the number of free symbols in the expression.
    All **weight**s are set to 1.0.
    
    **func** - A SymPy expression representing the function
    
    **num_symbols** - The number of different free symbols in the function
    
    **weight** - The weight for the function in the random selection
    
    :param path: The path to the JSON file containing the function list
    :return: 
    """
    global functions
    global weights
    var_regex = "x\\d+"
    with open(path, 'r') as f:
        functions_list = json.load(f)
        for func in functions_list:
            i = 0
            while regex.search(var_regex, func) is not None:
                func = regex.sub(var_regex, "a" + str(i), func, count=1)
                i += 1
            fn = sympify(func)
            functions.append(
                {
                    "func": fn,
                    "num_symbols": len(fn.free_symbols),
                    "weight": 1.0,
                }
            )
    w = 1.0 / float(len(functions))
    weights = np.array([w for _ in functions])

def random_function(function_library: list, weight_list: list, weighted=True):
    """
    Returns a random function from the **function_library**.
    :param function_library: A list of functions to choose from
    :param weight_list: A list of weights for the random selection
    :param weighted: If the random selection should be weighted or not
    :return: A function at random
    """
    if weighted:
        fn = function_library[np.random.choice(len(function_library), p=weight_list)]
        return fn
    fn = function_library[np.random.randint(len(function_library))]
    return fn

def standard_constant_chance(d):
    return (10.0 - d)/100.0
def standard_single_variable_chance(d):
    return (10.0 - d) / 20.0

def construct_random_scalar_function(_symbols: int | list, function_library: list, weight_list: list, weighted=True, max_depth=5,
                                      constant_chance=standard_constant_chance, constant_range=(-10,10),
                                      single_variable_chance=standard_single_variable_chance, complex_functions=False):
     """
     Constructs a random scalar function from recursively applying functions from the **function_library**.

     :param _symbols: A number of symbols for the function to have or a list of symbols
     :param function_library: A list of functions to choose from
     :param weight_list: A list of weights for the random selection
     :param weighted: If the random selection of functions used to construct the final function should be weighted
     :param max_depth: The maximum depth of the function to be constructed (i.e., a*(b*(c*(d*e))) has depth 4)
     :param constant_chance: The chance of a constant being used as an input, as a function of the maximum depth - the current depth
     :param constant_range: The range of constants to be used
     :param single_variable_chance: The chance of a single variable being used as an input, as a function of the maximum depth - the current depth
     :param complex_functions: If the constructed functions can be complex, or if only the real portion of it should be returned
     :return: A randomly constructed scalar function
     """
     if type(_symbols) is int:
         _symbols = symbols('x:'+str(symbols), real=not complex_functions)
     func = random_function(function_library, weight_list, weighted)
     inputs = []
     for i in range(func["num_symbols"]):
         if max_depth == 0 or np.random.rand() < single_variable_chance(max_depth):
             inputs.append(_symbols[np.random.randint(len(_symbols))])
         elif np.random.rand() < constant_chance(max_depth):
             inputs.append(round(np.random.uniform(*constant_range), 2))
         else:
             inputs.append(construct_random_scalar_function(_symbols, function_library, weight_list, weighted, max_depth-1, constant_chance, constant_range, single_variable_chance))
     f_new = func['func'].subs([(symbol, inputs[i]) for i, symbol in enumerate(func['func'].free_symbols)])
     if not complex_functions:
         return re(f_new)
     return f_new

def construct_random_scalar_function_set(num_functions: int, num_symbols: int, function_library: list, weight_list: list, weighted=True, max_depth=10,
                                      constant_chance=standard_constant_chance, constant_range=(-10,10),
                                      single_variable_chance=standard_single_variable_chance, complex_functions=False,
                                      overshoot=0.5, num_processes: int | None = None, chunk_size: int = 10000, max_wait=1.0) -> set:
    """
        Constructs a set of random scalar functions from recursively applying functions from the **function_library**.

        This method parallelizes **construct_random_scalar_function**.

        As this method eliminates duplicate functions, the number of returned functions may be less than *num_functions*.
        For this reason, the **overshoot** parameter will increase the number of functions returned to attempt to overshoot 
        the desired number of functions.
        
        Will print telemetry readouts to the terminal during processing.

        :param num_functions: The number of functions to be constructed
        :param num_symbols: A number of symbols for the function to have
        :param function_library: A list of functions to choose from
        :param weight_list: A list of weights for the random selection
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
    construct_fn_partial = functools.partial(construct_random_scalar_function, function_library=function_library,
                                              weight_list=weight_list, weighted=weighted, max_depth=max_depth,
                                              constant_chance=constant_chance, constant_range=constant_range,
                                              single_variable_chance=single_variable_chance,
                                              complex_functions=complex_functions)
    _symbols = symbols('x:' + str(num_symbols), real=not complex_functions)
    output = set()
    num_chunks = int(num_functions * (1.0 + overshoot) / chunk_size) + 1
    functions_per_chunk = int(num_functions * (1.0 + overshoot) / num_chunks)
    print(f"Constructing {functions_per_chunk * num_chunks} functions in {num_chunks} chunks")
    inputs = [_symbols for _ in range(functions_per_chunk)]
    start_time = datetime.datetime.now()
    for i in range(num_chunks):
        results = []
        print(f"Constructing a chunk of {functions_per_chunk} functions: {i + 1} / {num_chunks}")
        try:
            with ProcessPool(max_workers=num_processes if num_processes is not None else multiprocessing.cpu_count(),
                             max_tasks=1000) as pool:
                future = pool.map(construct_fn_partial, inputs, timeout=max_wait)
                iterator = future.result()
                for _ in tqdm(inputs, desc="Constructing Functions"):
                    try:
                        results.append(next(iterator))
                    except TimeoutError, ValueError, ArithmeticError, SyntaxError, ZeroDivisionError:
                        continue
                    except Exception as e:
                        print(f"\rError: {e}", flush=True)
                        traceback.print_exc()
        except:
            continue
        output |= set(results)
    end_time = datetime.datetime.now()
    print(f"Constructed {len(output)} functions.\n"
          f"Time elapsed: {(datetime.datetime(1, 1, 1) + (end_time - start_time)).strftime("%H:%M:%S")}")
    return output

def eval_function_at_point(function: Expr, point: list):
    """
    Attempts to evaluate a function at a given point by substituting in the given point value.

    Will return None if the function cannot be evaluated at the given point.

    :param function: The function to evaluate
    :param point: The point to evaluate the function at
    :return: The result of evaluating the function at the given point, or None if the function cannot be evaluated at the given point
    """
    try:
        r0 = function.evalf(subs={symbol: point[i] for i, symbol in enumerate(function.free_symbols)}, chop=True)
        if len(r0.free_symbols) > 0:
            f_str = str(function)
            for i, symbol in enumerate(function.free_symbols):
                f_str = f_str.replace(str(symbol), str(point[i]))
            r0 = sympify(f_str).evalf(chop=True)
        return r0
    except ValueError, ArithmeticError, SyntaxError, ZeroDivisionError, OverflowError, MemoryError, TypeError:
        return None
    except Exception as e:
        print(f"\rError: {e}", flush=True)
        traceback.print_exc()
        return None

def test_constant(function: Expr, num_test_points=10, tolerance=0.000001) -> bool:
    """
    Tests if a given function evaluates to a constant by testing it over several points.
    
    If the function returns the same value at all test points, it is considered constant.
    
    :param function: The function to be tested
    :param num_test_points: The number of test points to use
    :param tolerance: The maximum percent difference between two values to be considered equal (i.e., a tolerance of 0.001 will consider a difference of less than 0.1% to be equal)
    :return: If the tested function is constant
    """
    test_points = [float(n) for n in (np.linspace(-50.0 * np.random.random() - 50.0, 50.0 * np.random.random() + 50.0, num_test_points // 2))]
    test_points.extend([float(n) for n in (np.linspace(-0.5 * np.random.random() - 0.5, 0.5 * np.random.random() + 0.5, num_test_points // 2))])
    all_test_points = [test_points.copy() for _ in range(len(function.free_symbols))]
    for tp in all_test_points:
        np.random.shuffle(tp)
    r0 = None
    i = 0
    while r0 is None:
        if i >= len(all_test_points):
            return True
        p = [all_test_points[j][i] for j in range(len(function.free_symbols))]
        i += 1
        r0 = eval_function_at_point(function, p)
    r1 = eval_function_at_point(function, [all_test_points[j][i] for j in range(len(function.free_symbols))])
    if r0 != r1:
        return False
    while i < len(all_test_points):
        p = [all_test_points[j][i] for j in range(len(function.free_symbols))]
        i += 1
        try:
            r = eval_function_at_point(function, p)
            if r is None or r == nan:
                continue
            if r != r0:
                return False
            if abs(r - r0) > tolerance or (r0 != 0 and abs((r - r0) / r0) > tolerance):
                return False
        except TypeError, MemoryError, OverflowError:
            continue
        except Exception as e:
            print(f"\rError: {e}", flush=True)
            traceback.print_exc()
            continue
    return True

def prune_function(function: Expr, prune_constants=True, num_test_points=10, tolerance=0.000001, complex_functions=False, round_n=2) -> Expr | None:
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
        if prune_constants and len(function.free_symbols) == 0:
            return None
        func_new = simplify(function)
        if func_new == nan:
            return None
        if round_n is not None:
            func_new = func_new.xreplace({n: round(n, round_n) for n in func_new.atoms(Number)})
        if not complex_functions:
            return re(func_new)
        return func_new
    except ValueError:
        return None
    except Exception as e:
        print(f"\rError: {e}", flush=True)
        traceback.print_exc()
        return None

def prune_function_set(function_set: list | set, prune_constants=True, num_test_points=10, tolerance=0.000001, complex_functions=False, round_n: int | None = 2, num_processes: int | None = None, max_wait=1.0) -> set:
    """
        Simplifies and tests each function in a set, using certain criteria to determine if it should be discarded.
        
        This method parallelizes **prune_function**.
        
        Will print telemetry readouts to the terminal during processing.
        
        :param function_set: A list or set of functions to be simplified and tested
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
    print(f"Pruning {len(function_set)} functions")
    start_time = datetime.datetime.now()
    i = 0
    j = 0
    with ProcessPool(max_workers=num_processes if num_processes is not None else multiprocessing.cpu_count()) as pool:
        future = pool.map(prune_fn_partial, function_set, timeout=max_wait)
        iterator = future.result()
        for _ in tqdm(function_set, desc="Pruning"):
            try:
                n = next(iterator)
                if n is not None:
                    results.append(n)
                i += 1
            except TimeoutError, ValueError, ArithmeticError, SyntaxError, ZeroDivisionError, TypeError, OverflowError:
                continue
            except Exception as e:
                print(f"\rError: {e}", flush=True)
                traceback.print_exc()
                continue
    new_function_set = set(results)
    end_time = datetime.datetime.now()
    print(f"Pruned to {len(new_function_set)} functions.\n"
          f"Time elapsed: {(datetime.datetime(1, 1, 1) + (end_time - start_time)).strftime("%H:%M:%S")}")

    return new_function_set

ARITHMETIC_BIFUNCTIONS = [
    "(a + b)",
    "(a - b)",
    "(a * b)",
    "(a / b)"
]
def append_symbols(function: Basic | Expr | str | np.str_, num_symbols: int, bifunction_library=ARITHMETIC_BIFUNCTIONS):
    """
    Appends new symbols to a function until it has a given number of symbols.

    Uses a process of replacing a given variable *x* in the function with *f(a, b)* where
    *a* and *b* are variables not in the function and *f* is a bifunction from the **bifunction_library**.
    :param function: The function to append symbols to
    :param num_symbols: The total number of symbols the final function should have
    :param bifunction_library: A list of bifunctions as strings with symbols *a* and *b*
    :return: The function with symbols appended
    """
    free_symbols = set()
    str_function = ""
    if type(function) == str or type(function) == np.str_:
        str_function = function
        free_symbols = {f"x{i}" if regex.search(f"x{i}", str(function)) else None for i in range(num_symbols)}
        free_symbols.discard(None)
        if len(free_symbols) >= num_symbols:
            return sympify(str_function)
    else:
        if len(function.free_symbols) >= num_symbols:
            return function
        free_symbols = {str(symbol) for symbol in function.free_symbols}
        str_function = str(function)
    unused_symbols = {f"x{i}" for i in range(num_symbols)} - free_symbols
    while len(unused_symbols) > 0:
        try:
            x_pop: str = np.random.choice(list(free_symbols))
            free_symbols.remove(x_pop)
            unused_symbols.add(x_pop)

            x_add_1, x_add_2 = np.random.choice(list(unused_symbols), 2, replace=False)
            unused_symbols.remove(x_add_1)
            unused_symbols.remove(x_add_2)
            free_symbols.add(x_add_1)
            free_symbols.add(x_add_2)

            bifunc: str = np.random.choice(bifunction_library).copy()
            bifunc = bifunc.replace("a", x_add_1)
            bifunc = bifunc.replace("b", x_add_2)
            str_function = str_function.replace(x_pop, bifunc)
        except ValueError:
            continue
    return sympify(str_function)

def append_symbols_set(function_set: list | set, num_symbols: int, bifunction_library=ARITHMETIC_BIFUNCTIONS,
                      num_processes: int | None = None, max_wait=1.0):
    """
        Appends new symbols to all functions in a set until each has a given number of symbols.

        This method parallelizes **append_symbols**.

        Uses a process of replacing a given variable *x* in the function with *f(a, b)* where
        *a* and *b* are variables not in the function and *f* is a bifunction from the **bifunction_library**.

        Will print telemetry readouts to the terminal during processing.
        :param function_set: A list or set of functions to append symbols to
        :param num_symbols: The total number of symbols the final functions should have
        :param bifunction_library: A list of bifunctions as strings with symbols *a* and *b*
        :param num_processes: The maximum number of processes to use for parallelization, if None, will use all available processes
        :param max_wait: The maximum time to wait for a function to be appended to before moving on to the next one
        :return: The set of functions with symbols appended
    """
    append_symbols_partial = functools.partial(append_symbols, num_symbols=num_symbols, bifunction_library=bifunction_library)
    results = []
    print(f"Appending symbols to {len(function_set)} functions")
    start_time = datetime.datetime.now()
    with ProcessPool(max_workers=num_processes if num_processes is not None else multiprocessing.cpu_count()) as pool:
        future = pool.map(append_symbols_partial, function_set, timeout=max_wait)
        iterator = future.result()
        for _ in tqdm(function_set, desc="Appending Symbols"):
            try:
                results.append(next(iterator))
            except TimeoutError, ValueError, ArithmeticError, SyntaxError, ZeroDivisionError:
                continue
            except Exception as e:
                print(f"\rError: {e}", flush=True)
                traceback.print_exc()
    new_function_set = set(results)
    new_function_set.discard(None)
    end_time = datetime.datetime.now()
    print(f"Appended symbols to {len(new_function_set)} functions.\n"
          f"Time elapsed: {(datetime.datetime(1, 1, 1) + (end_time - start_time)).strftime("%H:%M:%S")}")

    return new_function_set

def generate_dataset(num_functions: int, directory_path: str, num_symbols: int, function_library: list, weight_list: list,
                     overshoot=0.5, weighted=True, max_depth=10,
                     constant_chance=standard_constant_chance, constant_range=(-10,10),
                     single_variable_chance=standard_single_variable_chance, complex_functions=False, 
                     prune_constants=True, num_test_points=10, tolerance=0.000001,
                     round_n: None | int = 3, append_missing_symbols=True, bifunction_library=ARITHMETIC_BIFUNCTIONS,
                     num_processes: int | None = None, max_generate_wait=1.0, max_prune_wait=1.0, max_append_wait=1.0):
    """
    Generates a dataset of random scalar functions of a certain size using **construct_random_scalar_function_set**
    and **prune_function_set**. This dataset is saved to a JSON file as a list of strings.

    The JSON file will be saved to **directory_path** as **n{num_functions}-x{symbols}-y1-d{max_depth}.json**.

    Two or more passes will be taken to fully generate the dataset.
    The first pass will generate the majority of the dataset but will intentionally undershoot the desired number of functions.
    The second pass will overshoot the remaining number of functions such that the dataset will be full after pruning.
    If the second pass does not fill the dataset, additional similar passes will be taken until the dataset is full.
    Any extraneous functions will be discarded.

    This method uses parallel processing to speed up the generation process.

    Will print telemetry readouts to the terminal during processing.

    :param num_functions: The number of functions to be generated to make up the dataset
    :param directory_path: The path to the directory where the dataset will be saved
    :param num_symbols: The number of symbols for the functions to have
    :param function_library: A list of functions to choose from
    :param weight_list: A list of weights for the random selection
    :param overshoot: A multiple of the desired number of functions to attempt to overshoot by: true_num_functions = num_functions * (1 + overshoot). This parameter is scaled by 2.5 in the second pass and onwards
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
    :param append_missing_symbols: If missing symbols should be appended with **append_symbols_set** after pruning
    :param bifunction_library: A list of bifunctions as strings with symbols *a* and *b* to be used by **append_symbols_set**
    :param num_processes: The maximum number of processes to use for parallelization, if None, will use all available processes
    :param max_generate_wait: The maximum time to wait for a function to be constructed before moving on to the next one
    :param max_prune_wait: The maximum time to wait for a pruned before moving on to the next one
    :param max_append_wait: The maximum time to wait for a function to be appended to before moving on to the next one
    :return: The generated dataset
    """
    start_time = datetime.datetime.now()
    dataset = construct_random_scalar_function_set(num_functions, num_symbols, function_library=function_library,
                                                   weight_list=weight_list, weighted=weighted, max_depth=max_depth,
                                                   constant_chance=constant_chance, constant_range=constant_range,
                                                   single_variable_chance=single_variable_chance, complex_functions=complex_functions,
                                                   overshoot=overshoot, num_processes=num_processes, max_wait=max_generate_wait)
    dataset = prune_function_set(dataset,
                                  prune_constants=prune_constants, num_test_points=num_test_points, tolerance=tolerance,
                                  complex_functions=complex_functions, round_n=round_n,
                                  max_wait=max_prune_wait, num_processes=num_processes)
    while len(dataset) < num_functions:
        temp_dataset = construct_random_scalar_function_set(num_functions - len(dataset), num_symbols,
                                                   weight_list=weight_list, function_library=function_library, weighted=weighted, max_depth=max_depth,
                                                   constant_chance=constant_chance, constant_range=constant_range,
                                                   single_variable_chance=single_variable_chance, complex_functions=complex_functions,
                                                   overshoot=2.5 * overshoot, num_processes=num_processes, max_wait=max_generate_wait)
        dataset |= prune_function_set(temp_dataset,
                                  prune_constants=prune_constants, num_test_points=num_test_points, tolerance=tolerance,
                                  complex_functions=complex_functions, round_n=round_n,
                                  max_wait=max_prune_wait, num_processes=num_processes)
    if append_missing_symbols:
        dataset = append_symbols_set(dataset, num_symbols, bifunction_library=bifunction_library,
                                    num_processes=num_processes, max_wait=max_append_wait)
    dataset = list(dataset)
    np.random.shuffle(dataset)
    dataset = dataset[:num_functions]
    print("\rDataset fully generated, saving to file")
    with open(f"{directory_path}/n{num_functions}-x{num_symbols}-y1-d{max_depth}.json", 'w') as f:
        json.dump([str(datum) for datum in dataset], f, indent=4)
    print(f"Dataset saved to {directory_path}/n{num_functions}-x{num_symbols}-y1-d{max_depth}.json")
    end_time = datetime.datetime.now()
    time_elapsed = (end_time - start_time)
    print(f"\nTotal Generation Time: {(datetime.datetime(1, 1, 1) + time_elapsed).strftime("%H:%M:%S")}"
          f"\nFunctions Generated per Second: {num_functions / time_elapsed.total_seconds():.2f}\n")
    return dataset

def randomize_function(function_library: list, num_symbols: int):
    """
    Produces a random function by randomizing the symbols in a random function from the given function library.
    :param function_library: A list of functions to choose from
    :param num_symbols: The number of symbols available for the function to have
    :return: A randomized function as a string
    """
    datum: str = np.random.choice(function_library)
    regex.sub("x\\d+", lambda m: "x" + str(np.random.randint(num_symbols)), datum)
    return datum

def extend_dataset(num_functions: int, dataset_path: str, directory_path: str, num_symbols: int,
                   overshoot=0.05, append_missing_symbols=True, bifunction_library=ARITHMETIC_BIFUNCTIONS,
                   num_processes: int | None = None, max_randomize_wait=1.0, max_append_wait=1.0):
    """
    Extends an existing dataset to more functions or more symbols or both by randomizing a given number
    of functions from that dataset using **randomize_function**.
    The expanded dataset is saved to a JSON file as a list of strings.

    The JSON file will be saved to **directory_path** as **n{num_functions}-x{symbols}-y1-ex.json**.

    This method uses parallel processing to speed up randomizing functions and appending missing symbols.

    Will print telemetry readouts to the terminal during processing.

    :param num_functions: The number of functions that will make up the extended dataset
    :param dataset_path: The file path to the dataset to be extended
    :param directory_path: The path to the directory where the extended dataset will be saved
    :param num_symbols: The number of symbols that each function in the extended dataset will have
    :param overshoot: A multiple of the desired number of functions to attempt to overshoot by: true_num_functions = num_functions * (1 + overshoot)
    :param append_missing_symbols: If missing symbols should be appended with **append_symbols_set** after pruning
    :param bifunction_library: A list of bifunctions as strings with symbols *a* and *b* to be used by **append_symbols_set**
    :param num_processes: The maximum number of processes to use for parallelization, if None, will use all available processes
    :param max_randomize_wait: The maximum time to wait for a function to be randomized before moving on to the next one
    :param max_append_wait: The maximum time to wait for a function to be appended to before moving on to the next one
    :return: The expanded dataset
    """
    start_time = datetime.datetime.now()
    dataset = set()
    true_num_functions = round(num_functions * (1.0 + overshoot))
    print(f"Extending dataset from {dataset_path}")
    with open(dataset_path, 'r') as f:
        temp_dataset = json.load(f)
        randomize_function_partial = functools.partial(randomize_function, num_symbols=num_symbols)
        results = []
        with ProcessPool(max_workers=num_processes if num_processes is not None else multiprocessing.cpu_count()) as pool:
            future = pool.map(randomize_function_partial, [temp_dataset for _ in range(true_num_functions)], timeout=max_randomize_wait)
            iterator = future.result()
            for _ in tqdm(range(true_num_functions), desc="Randomizing Functions"):
                try:
                    results.append(next(iterator))
                except TimeoutError, ValueError, ArithmeticError, SyntaxError, ZeroDivisionError:
                    continue
                except Exception as e:
                    print(f"\rError: {e}", flush=True)
                    traceback.print_exc()
    if append_missing_symbols:
        dataset = append_symbols_set(results, num_symbols, bifunction_library=bifunction_library,
                                    num_processes=num_processes, max_wait=max_append_wait)

    dataset = list(dataset)
    np.random.shuffle(dataset)
    dataset = dataset[:num_functions]
    print("\rDataset fully expanded, saving to file")
    with open(f"{directory_path}/n{num_functions}-x{num_symbols}-y1-ex.json", 'w') as f:
        json.dump([str(datum) for datum in dataset], f, indent=4)
    print(f"Dataset saved to {directory_path}/n{num_functions}-x{num_symbols}-y1-ex.json")
    end_time = datetime.datetime.now()
    time_elapsed = (end_time - start_time)
    print(f"\nTotal Expansion Time: {(datetime.datetime(1, 1, 1) + time_elapsed).strftime("%H:%M:%S")}"
          f"\nFunctions per Second: {num_functions / time_elapsed.total_seconds():.2f}\n")
    return dataset

def main():
    #extend_dataset(
    #    100000,
    #    "datasets/n1000-x5-y1-d5.json",
    #    "datasets",
    #    10
    #)
    load_functions_dict("datasets/standard_functions.json")
    results = parallel_processing.queued_parallel_processing(
        1000,
        construct_random_scalar_function,
        {
            "_symbols": list(symbols('x:5')),
            "function_library": functions,
            "weight_list": weights,
            "max_depth": 5,
        },
        [
            prune_function,
            append_symbols,
        ],
        [
            {},
            {"num_symbols": 5},
        ],
        json_path="datasets/n1000-x5-y1-d5.json",
        debug=True
    )
    #print(results)
    #generate_dataset(
    #    100000,
    #    "datasets",
    #    5,
    #    function_library=functions,
    #    weight_list=weights,
    #    weighted=True,
    #    max_depth=5,
    #)

if __name__ == "__main__":
    freeze_support()
    main()