#	NDP-h.py
#
#	Non-Deterministic Processor (NDP) - efficient parallel SAT-solver
#   In honor of Youcef Hamadache.
#	Copyright (c) 2024 GridSAT Stiftung
#
#	This program is free software: you can redistribute it and/or modify
#	it under the terms of the GNU Affero General Public License as published by
#	the Free Software Foundation, either version 3 of the License, or
#	(at your option) any later version.
#
#	This program is distributed in the hope that it will be useful,
#	but WITHOUT ANY WARRANTY; without even the implied warranty of
#	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#	GNU Affero General Public License for more details.
#
#	You should have received a copy of the GNU Affero General Public License
#	along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
#	GridSAT Stiftung - Georgstr. 11 - 30159 Hannover - Germany - ipfs: gridsat.eth/ - info@gridsat.io
#
"""
Non-Deterministic Processor (NDP-h)
====================================================
The Non-Deterministic Processor (NDP-h) is a tribute to Youcef Hamadache, devised as an
efficient parallel SAT-solver. This tool is exclusively tailored for efficiently solving
CNFs generated for Paul Purdom and Amr Sabry's Factoring Problems.

Parallel Breadth-First Search and Preparation for Parallel Iterative Processing:
--------------------------------------------------------------------------------
NDP-h initiates its resolution flow with a parallel CPU breadth-first search, dissecting the SAT problem
into smaller, independently solvable sub-formulas. This strategic decomposition into distinct sub-formulas
is pivotal for the NDP's efficiency, as it prepares the groundwork for further leveraging distributed
computing to tackle each sub-formula concurrently.

Parallel Processing with Ray:
------------------------------
Following the specific organization and prioritization of clauses, NDP-h employs Ray, a distributed
computing framework, for the parallel processing of the identified sub-formulas. This integration
enables NDP-h to distribute the computational workload across multiple CPU cores in a cluster,
significantly enhancing the NDP's capacity to address more complex SAT problems efficiently.

Method:
-------
At the core of NDP-h's approach is an exclusive implementation non-conform with the theory coded in
NDP-blueprint-thief available at https://github.com/GridSAT/NDP-blueprint-thief
on IPFS ipfs://QmUezbkQV2bPh3HzYbLe8YsgNHgaRC3a7cWKh2kiVYrdAQin and at http://gridsat.io.
The NDP-h method only organizes clauses by their length and their initial index, giving precedence
to unit clauses.

Unit Clauses:
-------------
Unit clauses comprising a single variable represent the bits of the product. The sorting criteria ensures unit clauses
to be processed first, thereby streamlining the resolution process by focusing on critical variables at the beginning
of the computation.

First Assignment:
-----------------
Unlike the comprehensive NDP-blueprint-thief that outputs all solutions by default (except opted otherwise), NDP-h 
always terminates upon finding the first satisfying assignment (provided the input is not prime) with no BDD output,
keeping the memory footprint low and almost constant during the whole resolution process.

Operational Dynamics:
---------------------
The compelling aspect of NDP-h emerges with the relationship between the overall problem size and the proportion
of new variables (VARs) required. While the absolute memory size and the number of variables tend to increase with
the complexity of the factorization challenges addressed, a nuanced efficiency is observed in the relative decrease
of the percentage of new VARs compared to the overall problem size. This efficiency is intrinsically linked to the
specific input format derived from Paul Purdom and Amr Sabry's CNF Generator for Factoring Problems, particularly
its asymmetric 3CNF transformation approach.

Conclusion:
-----------
The NDP-h approach to solving SAT problems pertaining to factorization challenges, demonstrates a significant
application of parallel processing techniques in computational number theory and cryptography. By integrating
a methodical decomposition of problems, employing unit clause prioritization and leveraging parallel computing
for resolution, NDP-h exemplifies an effective strategy for tackling complex SAT problems.
Its focus on terminating upon the first satisfying assignment underscores the solver's practical orientation
towards obtaining solutions efficiently, without delving into the exhaustive exploration of the potential solution
space with BDD generation.

Key Features:
------------
- Parses and validates DIMACS formatted files for Paul Purdom and Amr Sabry's CNF factorization challenges.
- Extracts essential metadata and prepares data for processing.
- Outputs processed data for use in further computation or analysis in .JASON, .txt, and .csv with input-, ID,
  as well as stopping-size percentage in file name for streamlined result organization.

Installation and Usage Guide:
============================

Prerequisites:
-------------
- Python 3.x
- pip
- virtualenv (optional for creating isolated Python environments)
- Ray (for distributed computing)


requirements.txt
----------------
ray[default]  # make a .txt file and copy into script directory 


Installation Steps:
------------------
1.	Start a screen session, e.g., screen -S NDP-h
2.	Make a directory, e.g., mkdir NDP-h
3.	Go to the directory, e.g., cd NDP-h
4.	Clone the Repository: git clone https://github.com/GridSAT/NDP-h.git (if published)
	or copy NDP-h.py and requirements.txt into the directory
5.	Install Python 3 package manager (pip) and sysstat: sudo apt install python3-pip sysstat
6.	Set Up a Virtual Environment (Optional): pip install virtualenv.
7.	Create and activate a virtual environment: virtualenv NDP-h   then   source NDP-h/bin/activate
8.	Install dependencies: pip install -r requirements.txt
9.	Make a directory for the inputs, e.g., mkdir NDP-h/inputs
10. Generate DIMACS files at Paul Purdom and Amr Sabry's CNF Generator at:
	https://cgi.luddy.indiana.edu/~sabry/cnf.html
	For bit-wise input generation, use e.g.: https://bigprimes.org/RSA-challenge
	
	or
	
	generate DIMACS locally with: https://github.com/GridSAT/CNF_FACT-MULT
	or on IPFS ipfs://QmYuzG46RnjhVXQj7sxficdRX2tUbzcTkSjZAKENMF5jba
	
11.	Copy the generated DIMACS files into the inputs directory.

Using Ray for Distributed Computing:
-----------------------------------
- Follow Ray documentation at https://docs.ray.io/ for setup instructions.
- Example commands for starting the head node (without ray dashboard) in a cluster with 4 worker nodes:

	Start head node without Ray Dashboard - example initialization with 4 CPUs as system reserves:
		export RAY_DISABLE_IMPORT_WARNING=1
		CPUS=$(( $(lscpu --online --parse=CPU | egrep -v '^#' | wc -l) - 4 ))
		ray start --head --include-dashboard=false --disable-usage-stats --num-cpus=$CPUS
		
	Start worker nodes - example initialization with 1 CPUs system reserves:
		export RAY_DISABLE_IMPORT_WARNING=1
		CPUS=$(( $(lscpu --online --parse=CPU | egrep -v '^#' | wc -l) - 1 ))
		ray start --address='MASTER-IP:6379' --redis-password='MASTER-PASSWORT' --num-cpus=$CPUS

Running NDP:
-----------
Execute from the command line, specifying the path to the DIMACS formatted input file and optionally the output directory, e.g.:

	python3 NDP-h.py inputs/rsaFACT-64bit.dimacs -d
	
	CLI options: --input_file_path="inputs/INPUT.dimacs" --output_dir="outputs/"
				
				Execution options
				=================
				-b Breath-First (BF) only outputting a result JSON
				-r Resume from BF only choosing the respective result JSON from a list
				-s Save BF JSON along the full NDP execution for later re-use
				
				BF-options
				==========
				-q define a Queue Size for BF
				-p any digit from 0 - 99% to specify % of VARs for BF
				-a absolute #VARs for BF
				-d default Queue Size setting next lower power of 2 of #CPUs (recommended)
				 
If BF-options are not specified via CLI you will be prompted to enter respective values
				
Additional Resources:
--------------------
Paul Purdom and Amr Sabry's CNF Generator for Factoring Problems required.
visit https://cgi.luddy.indiana.edu/~sabry/cnf.html to generate a respective problem
or visit https://github.com/GridSAT/CNF_FACT-MULT for the source code
or visit ipfs://QmYuzG46RnjhVXQj7sxficdRX2tUbzcTkSjZAKENMF5jba for the source code on IPFS

File stats:
----------
1700 net code lines
 216 comment lines
  99 KB file size
  
"""

import os
import time
import random
import logging
import argparse
import csv
import hashlib
import re  # Regular Expression module for string searching and manipulation
from collections import deque
import json

# Suppress deprecation warnings to keep the console output clean
os.environ['RAY_DEDUP_LOGS'] = '0'

# Disable Ray's log deduplication feature if needed
os.environ['PYTHONWARNINGS'] = "ignore::DeprecationWarning"

# Basic configuration for logging:
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import ray # Ray is a framework for distributed computing
import argparse
import sys
import math
import subprocess

### Version of NDP code
NDP_h_VERSION = "3.2.2"

#Revision History
"""
1.0.0	- NDP-c with single CPU Breath-First (BF) and single CPU Satisfy Iterative
1.1.0	- NDP-c with single CPU BF and Parallel CPU Satisfy Iterative
1.2.0	- NDP-c with single CPU BF and Parallel CPU Satisfy Iterative with Ray

2.0.0	- NDP-g with GPU BF and CPU Satisfy Iterative
2.1.0	- NDP-g with GPU BF and Parallel CPU Satisfy Iterative
2.1.0	- NDP-g with GPU BF and Parallel CPU Satisfy Iterative with Ray

3.0.0	- NDP-h with CPU Parallel BF and Parallel CPU Satisfy Iterative
3.1.0	- NDP-h with Ray Parallel BF and Ray Parallel Satisfy Iterative
3.1.1	- added -q CLI options for setting Queue Size for BF
3.1.2	- refined CLI logic with combinations of -q and with -a and -p and added default -d
3.1.3	- refined -q CLI logic with fall backs to respect power of 2 inputs
3.1.4	- added -s CLI to optionally save BF results
3.1.5	- enhanced CLI logic and removed combinations of -q and with -a and -p
3.2.0	- BF JSON filename includes #bits and queue-size
3.2.1	- BF JSONs are read-in with -r via choice from list
3.2.2	- default -d rounding down queue size to next lower power of 2 of #CPUs
""" 

# Print NDP-h version information
print(f'\n\nNDP-h Version {NDP_h_VERSION}')

def round_down_to_power_of_two(number):
	"""
	Ensure number is a power of two, rounding down if necessary.
	If the number is already a power of two, it's returned as is.
	Otherwise, it's rounded down to the nearest lower power of two.
	"""
	# Ensure the number is at least 1
	number = max(1, number)
	
	# Check if the number is already a power of two
	if is_power_of_two(number):
		return number
	else:
		return 2 ** math.floor(math.log2(number))

def is_power_of_two(n):
	"""
	Check if `n` is a power of two.
	"""
	return n > 0 and (n & (n - 1)) == 0
	
def convert_memory(memory_bytes):
	"""
	Convert memory size from bytes to appropriate units (TB, GB, MB, bytes).
	"""
	if memory_bytes == 'N/A':
		return memory_bytes
	if isinstance(memory_bytes, str):
		memory_bytes = int(memory_bytes)
	if memory_bytes >= 1099511627776:
		return f"{memory_bytes / 1099511627776:.1f} TB"
	elif memory_bytes >= 1073741824:
		return f"{memory_bytes / 1073741824:.1f} GB"
	elif memory_bytes >= 1048576:
		return f"{memory_bytes / 1048576:.1f} MB"
	else:
		return f"{memory_bytes} bytes"

def initialize_and_print_cluster_resources():
	"""
	Initialize Ray and print cluster and node resources information.
	"""
	if not ray.is_initialized():
		current_dir = os.path.dirname(os.path.abspath(__file__))
		# Set Ray's logging level to only show errors + exclude potentially large files
		ray.init(runtime_env={"working_dir": ".", "excludes": ["*.log", "__pycache__/", ".git/", "*.js", "*.json"]}, logging_level=logging.ERROR)
		print("Ray initialized.\n")
	cluster_resources = ray.cluster_resources()
	num_cpus = int(cluster_resources.get("CPU", 1))  # Defaults to 1 if not available
	num_gpus = int(cluster_resources.get("GPU", 0))  # Defaults to 0 if not available
	
	 # Convert memory resources to human-readable format
	for resource, value in cluster_resources.items():
		if resource.endswith('memory'):
			cluster_resources[resource] = convert_memory(int(value))

	# Get information about all nodes in the cluster
	all_nodes = ray.nodes()

	# Create a string to store cluster resources information
	cluster_resources_info = "Cluster Resources:\n\n"
	for resource, value in cluster_resources.items():
		# Convert CPU count to integer if it's CPU resource
		if resource == 'CPU':
			value = int(value)
		else:
			value = int(value) if isinstance(value, float) else value
		cluster_resources_info += f"{resource}: {value}\n"

	# Iterate over each node and append its resources information to cluster_resources_info
	cluster_resources_info += "\n\nNode Resources:\n\n"
	for node in all_nodes:
		node_id = node['NodeID']
		cluster_resources_info += f"Node: {node_id}\n"
		
		# Handle CPU resource
		cpu_value = node['Resources'].get('CPU', 'N/A')
		if cpu_value != 'N/A':
			cpu_info = int(cpu_value)
		else:
			cpu_info = cpu_value
		cluster_resources_info += f"CPU: {cpu_info}\n"
		
		# Handle memory resource
		memory_bytes = node['Resources'].get('memory', 'N/A')
		memory_readable = convert_memory(memory_bytes) if memory_bytes != 'N/A' else 'N/A'
		cluster_resources_info += f"Memory: {memory_readable}\n"

		if 'GPU' in node['Resources']:
			cluster_resources_info += f"GPU: {int(node['Resources']['GPU'])}\n"
		else:
			cluster_resources_info += "No GPUs available on this node.\n"

		cluster_resources_info += "=" * 30 + "\n"

	# Iterate over each node and print its resources
	for node in all_nodes:
		print(f"Node: {node['NodeID']}")
		cpu_value = node['Resources'].get('CPU', 'N/A')
		if cpu_value != 'N/A':
			try:
				cpu_info = int(cpu_value)
			except ValueError:
				cpu_info = cpu_value
		else:
			cpu_info = cpu_value
		print(f"CPU: {cpu_info}")
		
		# Handle memory resource
		memory_bytes = node['Resources'].get('memory', 'N/A')
		if memory_bytes != 'N/A':
			memory_readable = convert_memory(int(memory_bytes))
		else:
			memory_readable = memory_bytes
		print(f"Memory: {memory_readable}")

		if 'GPU' in node['Resources']:
			print(f"GPU: {int(node['Resources']['GPU'])}")
		else:
			print("No GPUs available on this node.")
		print("=" * 30)

	return cluster_resources, num_cpus, num_gpus, cluster_resources_info, memory_readable

def generate_problem_id(problem_data, start_timestamp, cluster_resources):
	"""
    Generates a unique problem ID based on problem data, cluster resources, and the start timestamp.

    Parameters:
        problem_data (dict): A dictionary containing problem-specific data.
        cluster_resources (dict): A dictionary of the resources available in the Ray cluster.
        start_timestamp (str): The start timestamp of the problem-solving session.

    Returns:
        str: A unique MD5 hash serving as the problem ID.
    """
	current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
	problem_data.update({
		"Start Timestamp": start_timestamp,
		"Generation Timestamp": current_time,
		"Cluster Resources": cluster_resources
	})
	data_str = ""
	for key, value in sorted(problem_data.items()):
		if isinstance(value, dict):
			value_str = ",".join([f"{k}:{v}" for k, v in sorted(value.items())])
			data_str += f"{key}:{value_str},"
		else:
			data_str += f"{key}:{value},"
	problem_id = hashlib.md5(data_str.encode()).hexdigest()
	return problem_id

def read_input_files(file_path, output_dir=None):
	"""
    Processes the input file specified by the file_path, extracting important SAT problem-related information.

    Parameters:
        file_path (str): The path to the input file containing the DIMACS CNF formula and other metadata.
        output_dir (str, optional): The directory where output files will be saved. If not provided, uses the current directory.

    Returns:
        tuple: A tuple containing the DIMACS formula, input digit, variables for both inputs, number of clauses, file name, and output file names.
    
    Raises:
        ValueError: If required header information is missing or the input file is in an incorrect format.
    """
	# Initialize variables to store header information
	dimacs_formula = None
	input_digit = None
	first_input_vars = None
	second_input_vars = None
	num_clauses = 0	 # Variable to count the number of clauses

	with open(file_path, 'r') as file:
		try:
			file_name = os.path.basename(file_path)  # Get the file name
			lines = file.readlines()
			output_file_name_txt = None
			output_file_name_csv = None
			print("\nreading input file:", file_name)

			# Check if the file contains the required header information
			for line in lines:
				if line.startswith("c Circuit for product"):
					input_digit = int(re.search(r'\d+', line).group())
				elif line.startswith("c Variables for first input"):
					first_input_vars = [int(x) for x in re.findall(r'\d+', line)]
				elif line.startswith("c Variables for second input"):
					second_input_vars = [int(x) for x in re.findall(r'\d+', line)]
				elif line.startswith("p cnf"):
					dimacs_formula = "".join(lines[lines.index(line):])
					num_clauses = sum(1 for line in lines if not line.startswith("c") and not line.startswith("p"))
					break

				# Check if all required information is extracted
				if dimacs_formula is not None and input_digit is not None and first_input_vars is not None and second_input_vars is not None:
					break

		except ValueError as e:
			# Print only the error message without the traceback
			print(e)
			return
			
	# Check if all required information is extracted
	if dimacs_formula is None or input_digit is None or first_input_vars is None or second_input_vars is None:
		print(f"\n")
		raise ValueError("\n\nPaul Purdom and Amr Sabry header information missing or incorrect format in the input file.\n\nCNF Generator for Factoring Problems required.\n\nvisit https://cgi.luddy.indiana.edu/~sabry/cnf.html to generate a respective problem\nor\nvisit https://github.com/GridSAT/CNF_FACT-MULT for the source code\nor\nvisit ipfs://QmeeDsP4WnHG5F8dYazDnyUGhdnF1t8J4i4YAMgXCa6wB4 for the source code on IPFS\n\n")
	print(f"                    input file OK\n")
	return dimacs_formula, input_digit, first_input_vars, second_input_vars, num_clauses, file_name, output_file_name_txt, output_file_name_csv

	"""
	This section includes utility functions for processing SAT problem instances and their solutions. These functions are essential for the analysis and transformation of the problem data, aiding in the broader task of finding satisfying assignments or demonstrating unsatisfiability.
	
	- non_zero_subarrays_recursive: Recursively navigates through a nested list structure to extract all non-zero subarrays. This function is critical for filtering out irrelevant data and focusing on meaningful results, particularly after applying resolution steps or when analyzing the final output of the SAT solving process.
	
	- replace_elements: Matches elements between two lists based on the absolute value of their integers, replacing elements in the target lists with those found in a source array. This function is vital for reconciling the SAT solver's output with specific variable assignments, ensuring the results are correctly interpreted in the context of the original problem.
	
	- flatten_list_of_lists: Converts a nested list structure into a flat list. This simplification is necessary for various processing stages, such as when preparing data for output or further analysis, ensuring that complex nested results are accessible and manageable.
	
	- convert_to_binary_array_and_integer: Transforms a list of integers into a binary representation and then computes its corresponding integer value. This function is crucial for interpreting variable assignments as binary numbers, facilitating the evaluation of SAT solutions in numerical contexts, such as verifying the factors of an integer in cryptographic applications.
	"""
def parse_dimacs(dimacs_formula):
	# Split the formula into lines and remove empty lines
	lines = [line.strip() for line in dimacs_formula.split('\n') if line.strip()]
	
	# Extract the number of variables and clauses from the 'p cnf' line
	_, _, num_vars, num_clauses = lines[0].split()
	num_vars, num_clauses = int(num_vars), int(num_clauses)
	
	# Initialize the formula array
	formula = []

	# Fill in the formula based on the clauses
	for line in lines[1:]:
		literals = [int(lit) for lit in line.split()[:-1]]
		formula.append(literals)

	# For all clauses which have less than 3 literals: Add zeros
	for clause in formula:
		clause.extend([0] * (3 - len(clause)))

	return formula

def ResolutionStep(A, i):
	"""
    Performs the resolution step on a given formula A with respect to a chosen literal i.

    This function applies resolution to simplify the formula based on the selected literal i.
    It generates two sets of clauses (sub-formulas): one assuming the literal i is true (Left Array),
    and the other assuming it is false (Right Array).

    Parameters:
    - A: The current set of clauses (formula) represented as a list of lists where each sub-list is a clause.
    - i: The chosen literal for the resolution step.

    Returns:
    - LA: The modified set of clauses assuming literal i is true.
    - RA: The modified set of clauses assuming literal i is false.
    """
	LA = []
	RA = []
	for subarray in A:
		modified_subarray = [x + i for x in subarray]
		if 2*i in modified_subarray:
			continue
		LA.append(modified_subarray)
	LA = [[x - i if x != 0 else x for x in subarray] for subarray in LA]
	for subarray in A:
		modified_subarray = [x - i for x in subarray]
		if (-1)*2*i in modified_subarray:
			continue
		RA.append(modified_subarray)
	RA = [[x + i if x != 0 else x for x in subarray] for subarray in RA]
	return LA, RA

def choice(A):
	"""
    Selects a variable from the given 2D array 'A' for the next resolution step based on specific criteria.

    The selection criteria are as follows:
    1. If a subarray contains exactly two zeros, the function returns the absolute value of the non-zero integer.
    2. If no subarray meets the first criteria, and there's a subarray with one zero and two integers, 
       it returns the absolute value of the first non-zero integer.
    3. If neither condition is met, it returns the absolute value of the first integer in the first subarray.

    Parameters:
        A (list of list of int): The 2D array of integers from which to select the next variable.

    Returns:
        int: The absolute value of the selected integer for the next resolution step.
    """
	# Check for subarray with two zeros
	for subarray in A:
		if subarray.count(0) == 2:
			# Choose the non-zero integer in the subarray and return its absolute value
			return abs(next(x for x in subarray if x != 0))

	# Check for subarray with one zero and two integers
	for subarray in A:
		if subarray.count(0) == 1 and len(subarray) == 3:
			# Choose the first occurring non-zero integer in the subarray and return its absolute value
			return abs(next(x for x in subarray if x != 0))

	# If no subarrays with zeros, choose the first integer in the first subarray and return its absolute value
	return abs(A[0][0]) if A and A[0] else None

def Satisfy_iterative_two_arguments(A, assignment):
	"""
    Wrapper function that calls the Satisfy_iterative function with the given 2D array 'A' 
    and returns its result along with the original 'assignment'.

    This function is designed to integrate Satisfy_iterative into workflows that require maintaining 
    the original assignment alongside the result of the satisfiability check.

    Parameters:
        A (list of list of int): The 2D array representing clauses in a SAT problem.
        assignment (list of int): The original assignment (contextual data) that is passed through unchanged.

    Returns:
        tuple: A tuple where the first element is the result of Satisfy_iterative function and 
               the second element is the original 'assignment'.
    """
	# Returns the result of Satisfy_iterative and the unchanged assignment
	result = Satisfy_iterative(A)
	return result, assignment

def Satisfy_iterative(A):
	"""
    Iteratively attempts to find a satisfying assignment for the formula A.

    It uses a stack to manage branching with chosen literals and applies resolution steps
    to simplify the formula. It explores both possibilities for each chosen literal (true and false)
    until a satisfying assignment is found or all options are exhausted.

    Parameters:
    - A: The initial set of clauses (formula) to be satisfied, represented as a list of lists.

    Returns:
    - A list of satisfying assignments, if any, or an indication of unsatisfiability.
    """
	# Iteratively finds satisfying assignments
	stack = [(A, [])]  # Initialize stack with the initial formula and empty choices
	results = []  # Store the results
	result_LA = []	# Store the left assignment
	result_RA = []	# Store the right assignment
	while stack:  # Iterate until stack is empty
		current_A, choices = stack.pop()  # Pop the current formula and choices
		i = choice(current_A)  # Choose a variable index from the current formula
		if i is None:  # If no more variables are available
			results.append(choices)	 # Append the current choices to results
			continue

		LA, RA = ResolutionStep(current_A, i)  # Apply resolution step
		if not LA or any(subarray == [0, 0, 0] for subarray in LA):	 # Check for unsatisfiability in left assignment
			if not LA:
				result_LA = choices + [i]  # Set left assignment result if LA is empty
			else:
				result_LA = [0]	 # Set left assignment result to unsatisfiable
		else:
			stack.append((LA, choices + [i]))  # Add new formula and updated choices to stack

		if not RA or any(subarray == [0, 0, 0] for subarray in RA):	 # Check for unsatisfiability in right assignment
			if not RA:
				result_RA = choices + [-i]	# Set right assignment result if RA is empty
			else:
				result_RA = [0]	 # Set right assignment result to unsatisfiable
		else:
			stack.append((RA, choices + [-i]))	# Add new formula and updated choices to stack

		results.append([result_LA, result_RA])	# Append left and right assignment results to results list

	return results	# Return the satisfying assignments

@ray.remote
def process_task_ray(task):
	"""
	Process a task in a Ray remote function, involving resolution steps and comparison of 2D arrays.
	"""
	current_A, current_choices, length = task
	i = choice(current_A)

	if i is None:
		return None, None, length

	LA, RA = ResolutionStep(current_A, i)
	new_tasks = []

	if not contains_zero_array(LA):
		new_tasks.append((LA, current_choices + [i], len(LA)))

	if not contains_zero_array(RA):
		diff, equal_cond = compare_2D_arrays(LA, RA)
		if not equal_cond:
			new_tasks.append((RA, current_choices + [-i], len(RA)))

	return new_tasks

def satisfy_breadth_first_parallel_ray(A, sizeOfCNF, iterations, num_vars, percentage=None, absolute=None, queue_size=None, default_size=False):
	"""Parallel breadth-first search algorithm using Ray.

	This function conducts a parallel breadth-first search using Ray for efficient processing. It iteratively explores various configurations
	to satisfy given constraints or until a stopping condition is met.
	
	Parameters:
	
		A: Initial array configuration.
		sizeOfCNF: Size of the CNF formula.
		iterations: Maximum number of iterations.
		num_vars: Number of variables.
		percentage: Percentage of variables to consider as a stopping criterion.
		absolute: Absolute number of variables to consider as a stopping criterion.
		queue_size: Maximum size of the processing queue.
		default_size: Flag indicating whether to use default queue size.
	
	Returns:
	
		queue: The final processing queue.
		queue_length: Length of the final processing queue.
		clause_sets_list: List of clause sets explored.
		assignments_list: List of variable assignments explored.
		factor: The stopping size or factor.
		factor_source: Source of the stopping size (percentage, absolute, queue_size).
		queue_size: The specified queue size.
		iteration_count: Total number of iterations processed.
		default_size: Flag indicating whether default queue size was used.
	"""
	cluster_resources = ray.cluster_resources()
	num_cpus = int(cluster_resources.get("CPU", 1))

	queue = deque([(A, [], sizeOfCNF)])

	print("\nBreadth-first parallel processing initiated..\n")
	print(f"                 CPUs: {int(num_cpus)}")
	print(f"                 VARs: {num_vars}\n")

	factor = None
	factor_source = None
	iterations = 0
	i = 0
	iteration_count = 0	 # Initialize iteration_count at the start
	
	if percentage is not None:
		factor = round(percentage * num_vars / 100)
		factor_source = "percentage"
# 		i = factor

		if 0 <= percentage <= 99.00:
			print(f"        Stopping size: {factor} ({percentage:.2f}%)\n")
			
			while i > 0 and queue:
				task_refs = [process_task_ray.remote(task) for task in queue]
				queue.clear()
	
				# Asynchronously gather results from Ray tasks
				for new_tasks in ray.get(task_refs):
					if new_tasks:
						queue.extend(new_tasks)
				i -= 1
				print(f" Remaining VARs: {i}, queue size: {len(queue)}")
	
		else:
			print("\nMax percentage value is 99% and must be positive - try again.\n")
			sys.exit(1)

	if absolute is not None:
		factor_source = "absolute"
		factor = absolute
# 		i = factor
		print(f"        Stopping size: {factor}\n")

		if absolute > 0 and absolute == int(absolute) and absolute < num_vars:
			
			while i > 0 and queue:
				task_refs = [process_task_ray.remote(task) for task in queue]
				queue.clear()
	
				# Asynchronously gather results from Ray tasks
				for new_tasks in ray.get(task_refs):
					if new_tasks:
						queue.extend(new_tasks)
				i -= 1
				print(f" Remaining VARs: {i}, queue size: {len(queue)}")

		else:
			# If absolute is not a positive whole number, has decimals, or is not less than num_vars
			if absolute >= num_vars:
				print(f"\n		 Absolute number must be smaller (num_vars) - retry.\n")
			else:
				print("		 Value must be a positive, whole number without decimals - retry.\n")
			sys.exit(1)
			
	if queue_size is not None:
		try:
			factor_source = "queue_size"
			print(f"           Queue size: {queue_size}\n")
			
			if queue_size > num_vars:
				print(f"           Queue size cannot exceed {num_vars} VARs - retry.\n")
				sys.exit(1)
	
			elif not is_power_of_two(queue_size):  # Check if queue_size is NOT a power of two
				print(f"           Adjust queue size to power of 2 - example: {round_down_to_power_of_two(num_cpus)}\n")
				sys.exit(1)
	
			while len(queue) <= queue_size:
				task_refs = [process_task_ray.remote(task) for task in queue]
				queue.clear()
		
				# Asynchronously gather results from Ray tasks
				for new_tasks in ray.get(task_refs):
					if new_tasks:
						queue.extend(new_tasks)
						
				iteration_count += 1
				print(f" Processed VARs: {iteration_count}, queue size: {len(queue)}")
	
				if len(queue) == queue_size:
					print(f"\n           Queue size limit reached ({queue_size}). Stopping..")
					break
	
				if iteration_count == num_vars -1:
					print(f"\n            #VARs cannot exceed {num_vars} VARs. Stopping..")
					break
					
		except ValueError:
			print(f"\nQueue size must be positive number and to the power of 2 - example: {round_down_to_power_of_two(num_cpus)}\n")
			sys.exit(1)

	if default_size is not False:
		# Set the queue_size to the next power of 2 of the number of CPUs
		queue_size = round_down_to_power_of_two(num_cpus)
		print(f"           Next lower power of 2 of {int(num_cpus)}: CPUs\n")
		print(f"           Queue size: {queue_size}\n")
		while len(queue) <= queue_size:
			task_refs = [process_task_ray.remote(task) for task in queue]
			queue.clear()

			# Asynchronously gather results from Ray tasks
			for new_tasks in ray.get(task_refs):
				if new_tasks:
					queue.extend(new_tasks)
					
			iteration_count += 1
			print(f" Processed VARs: {iteration_count}, queue size: {len(queue)}")

			if len(queue) >= queue_size:
				factor_source = "queue_size"
				print(f"\n           Queue size limit reached ({queue_size}). Stopping..")
				break
			if iteration_count == num_vars - 1:
				print(f"\n            #VARs cannot exceed {num_vars} VARs. Stopping..")
				break
			
		print(f"\n     Final queue size: {len(queue)}\n")

	# Extract clause sets and assignments
	clause_sets_list = [task[0] for task in queue]
	assignments_list = [task[1] for task in queue]
	
	# Calculate percentage_stats_q
	if queue_size == 0:
		percentage_stats_q = 0

	else:
		percentage_stats_q = (iteration_count / num_vars) * 100
	
	# Calculate percentage_stats_a
	if absolute is not None:
		percentage_stats_a = (int(absolute) / num_vars) * 100

	else:
		percentage_stats_a = 0  # Or any default value you prefer

		
	return queue, len(queue), clause_sets_list, assignments_list, factor, factor_source, queue_size, iteration_count, default_size, percentage_stats_q, percentage_stats_a

def save_bf_results(clause_sets_list, assignments_list, output_file_name_json):
	"""
	Save breadth-first search results to a JSON file.

	Parameters:
	- clause_sets_list: List of clause sets explored.
	- assignments_list: List of variable assignments explored.
	- output_path: Path to save the results.

	Raises:
	- ValueError: If output_path is None.

	"""
	if output_file_name_json is None:
		raise ValueError("\nOutput path is None. Please specify a valid path.\n")
	bfs_data = {
		"clauseSets": clause_sets_list,
		"assignments": assignments_list
	}
	with open(output_file_name_json, 'w') as file:
		json.dump(bfs_data, file, indent=4)

def list_available_json_files(output_dir):
	"""
	List available JSON files in the specified directory.

	Parameters:
	- directory: Directory path to search for JSON files.

	Returns:
	- List of available JSON file names.
	"""
	json_files = [f for f in os.listdir(output_dir) if f.endswith('.json')]
	return json_files

def select_json_file(json_files):
	"""
	Prompt the user to select a JSON file from a list.

	Parameters:
	- json_files: List of available JSON files.

	Returns:
	- selected_json_file: Filename of the selected JSON file, or None if no valid selection is made.
	"""
	print(f"\n\n       +++ satisfy parallel only +++\n")
	print("\nAvailable JSON files:\n")
	for i, json_file in enumerate(json_files, start=1):
		print(f"{i}. {json_file}")
	
	selection = input("\nEnter the number of the JSON file you want to load: ")
	try:
		selection_index = int(selection) - 1
		if 0 <= selection_index < len(json_files):
			return json_files[selection_index]
		else:
			print("\nInvalid selection. Please enter a number corresponding to a JSON file.\n")
			return None
	except ValueError:
		print("\nInvalid input. Please enter a number.\n")
		return None

def load_bf_results(output_dir, num_cores):
	"""
	Load breadth-first search results from a selected JSON file.

	Parameters:
	- directory: Directory path to search for JSON files.
	- num_cores: Number of CPU cores for parallel processing.

	Returns:
	- clause_sets: List of clause sets loaded.
	- assignments: List of variable assignments loaded.

	Raises:
	- FileNotFoundError: If no JSON files are found in the directory.
	"""
	json_files = list_available_json_files(output_dir)
	if not json_files:
		raise FileNotFoundError("\nNo JSON files found in the specified directory.\n")

	selected_json_file = select_json_file(json_files)
	if selected_json_file is None:
		print("\nNo JSON file selected. Exiting.\n")
		sys.exit(1)

	input_path = os.path.join(output_dir, selected_json_file)

	try:
		with open(input_path, 'r') as file:
			bfs_data = json.load(file)
		print("\n						    loaded.")
		print(f"\n\nevaluating {int(num_cores)} CPU cores for parallel processing..\n")
		return bfs_data["clauseSets"], bfs_data["assignments"]
		
	except FileNotFoundError:
		print("\nSelected BF results file not found. Please run with -b first.\n")
		sys.exit(1)

@ray.remote
def satisfy_parallel(clauseSet, assignment):
	"""
    Parallelized function using Ray.

    This function takes a clause set and an initial assignment, attempts to find a
    satisfying assignment for the clause set, and returns the result. It is designed
    to run in parallel across multiple CPU cores using Ray, a framework for distributed computing.

    Parameters:
    - clauseSet: A set of clauses representing a SAT problem to solve.
    - assignment: An initial assignment of values to variables for the SAT problem.

    Returns:
    - nonZeroResult + oldAssignment: The satisfying assignment if one is found, otherwise an empty list.
      The result combines the new satisfying assignment (nonZeroResult) with the initial assignment (oldAssignment)
      provided as input.

    Note:
    - The function prints 'processing' at the beginning to indicate the start of a task and 'completed' when
      a satisfying assignment is found or if the task concludes without finding one.
    - It utilizes 'Satisfy_iterative_two_arguments' to find satisfying assignments and 'non_zero_subarrays_recursive'
      to filter out non-zero subarrays, which are crucial for determining the SAT problem's solution.
    - This task is specifically allocated a single CPU core (`num_cpus=1`), indicating its execution in a separate
      process within Ray's distributed system.
    """
	print(f"processing")
	task_id = ray.get_runtime_context().get_task_id()
	# Compute resultAssignment and oldAssignment using Satisfy_iterative_two_arguments
	resultAssignment, oldAssignment = Satisfy_iterative_two_arguments(clauseSet, assignment)
	# Check if resultAssignment is not None
	if resultAssignment is not None:
		# Find non-zero subarrays recursively
		nonZeroResult = non_zero_subarrays_recursive(resultAssignment)
		# If nonZeroResult is not empty, return concatenated with oldAssignment
		if nonZeroResult != []:
			print(f"completed.")
			return nonZeroResult + oldAssignment
	return []  # Return empty list if no non-zero subarrays found
	
def parallel_satisfy_ray(clauseSets, assignments):
	"""
    Executes the SAT problem solving in parallel using Ray.

    This function takes a list of clause sets and their corresponding assignments,
    submits each pair as a separate task to be processed in parallel, and collects
    the results. It leverages Ray's distributed computing capabilities to efficiently
    manage task execution across available CPU cores.

    Parameters:
    - clauseSets: A list of clause sets, where each clause set represents a distinct SAT problem.
    - assignments: A list of initial assignments corresponding to each clause set.

    Returns:
    - endResult: A list of non-null results from the SAT problem solving tasks, indicating successful computations.
    - tasks_submitted: The total number of tasks submitted for execution, representing the workload size.
    - tasks_completed: The number of tasks that successfully completed and returned a non-null result.
    """
    
	# Submit each problem instance (clause set and assignment pair) to Ray for parallel processing.
    # Each 'satisfy_parallel' task will attempt to find a satisfying assignment for its clause set.
	futures = [satisfy_parallel.remote(clauseSet, assignment) for clauseSet, assignment in zip(clauseSets, assignments)]
			
	# Assuming each task could use 1 CPU, the number of active workers is the number of submitted tasks
	active_workers = len(futures)
	print(f"parallel processing using {active_workers}    workers:\n")			

	# Wait for the tasks to complete and collect the results
	results = ray.get(futures)

	endResult = [result for result in results if result]
	print(f"                              all done.")
	for result in results:
		if result:
			endResult.extend(result)
			break
	ray.shutdown() 	# Shutdown Ray
	return endResult, len(futures)

def discardFalse(B):
	"""
    Filters out clause sets containing the clause [0, 0, 0] from a collection of clause sets.

    The presence of [0, 0, 0] in a clause set indicates a contradiction or an unsatisfiable
    clause, making the entire clause set unsatisfiable. This function removes such clause sets
    to streamline the SAT solving process.

    Parameters:
    - B: A list of clause sets, where each clause set is a list of clauses.

    Returns:
    - A filtered list of clause sets, excluding any that contain the clause [0, 0, 0].
    """
	# Filter out 2D subarrays containing [0, 0, 0]
	filtered_subarrays = [subarray for subarray in B if not any(subarray_elem == [0, 0, 0] for subarray_elem in subarray)]
	return filtered_subarrays

def contains_zero_array(arr):
	"""
    Determines if any clause within a clause set is [0, 0, 0].

    A clause of [0, 0, 0] indicates an unsatisfiable condition within the clause set.
    Identifying such clauses can help in early detection of unsatisfiable clause sets.

    Parameters:
    - arr: A clause set, represented as a list of clauses.

    Returns:
    - True if [0, 0, 0] is found within the clause set, False otherwise.
    """
	return any(subarray == [0, 0, 0] for subarray in arr)

def compare_2D_arrays(L, R):
	"""
    Compares two clause sets to determine if there are any differences between them.

    This function is useful for identifying whether two different approaches or modifications
    to a clause set result in equivalent clause sets, indicating that no significant change
    was made by the modifications.

    Parameters:
    - L: The first clause set to compare.
    - R: The second clause set to compare.

    Returns:
    - A tuple containing two elements:
      1. A pair (diff_in_L, diff_in_R) indicating clauses present in one clause set but not the other.
      2. A boolean value, False if there are differences, True if the clause sets are equivalent.
    """
	# Find 1D arrays that are in L but not in R, or in R but not in L
	diff_in_L = [subarray for subarray in L if subarray not in R]
	diff_in_R = [subarray for subarray in R if subarray not in L]

	# Check if there are differences
	if diff_in_L or diff_in_R:
		return (diff_in_L, diff_in_R), False
	else:
		return None, True

def non_zero_subarrays_recursive(A):
	"""
    Recursively extracts non-zero subarrays or elements from a nested list structure.

    This function iterates through each item in the input list. If an item is a list,
    it recursively searches that sublist for non-zero subarrays or elements. If an item
    is not a list and is non-zero (neither the integer 0 nor a list containing only 0),
    it is considered a valid non-zero subarray (or element) and added to the result list.
    
    Parameters:
    - A (list): A nested list of integers or sublists. This list may contain zeros,
      non-zero integers, and further nested lists.

    Returns:
    - list: A flattened list containing all non-zero elements or subarrays extracted
      from the input list. This includes individual non-zero integers and non-zero
      sublists found at any depth within the nested list structure.
    """
	# Recursively extracts non-zero subarrays from a nested list
	result = []
	for subarray in A:
		if isinstance(subarray, list):
			# Recursively call the function for nested subarrays
			result.extend(non_zero_subarrays_recursive(subarray))
		elif ((subarray != [0]) and (subarray != 0)):
			# Append non-zero elements to the result list
			result.append(subarray)
	return result

def replace_elements(A, L1, L2):
	"""
	Replaces elements in two lists of sublists, L1 and L2, with elements from another list, A, based on their absolute values.
	
	Parameters:
	- A (list): A list of integers from which replacement elements are selected.
	- L1 (list of lists): A list containing sublists of integers. Elements in these sublists are candidates for replacement.
	- L2 (list of lists): Another list containing sublists of integers, similar to L1, where elements may be replaced.
	
	Returns:
	- tuple: A tuple containing two elements, where the first element is the modified L1 and the second element is the modified L2. Both L1 and L2 have had their elements replaced by elements from A, based on matching absolute values.
	"""
	# Iterate over each sublist in L1
	for i, sublist in enumerate(L1):
		for j, item in enumerate(sublist):
			# Check each element in A for a match
			for a in A:
				if abs(a) == abs(item):
					L1[i][j] = a # Replace the element in the sublist
					break

	# Repeat the process for L2
	for i, sublist in enumerate(L2):
		for j, item in enumerate(sublist):
			# Check each element in A for a match
			for a in A:
				if abs(a) == abs(item):
					L2[i][j] = a  # Replace the element in the sublist
					break
	return L1, L2

def flatten_list_of_lists(lst):
	"""
	Flattens a list of lists into a single list, merging all sublists into one list containing all the elements in the original sublists.
	
	Parameters:
	- lst (list of lists): A list containing other lists as its elements.
	
	Returns:
	- list: A single list containing all the elements from the sublists in the order they appeared in the original list of lists.
	"""
	# Flattens a list of lists into a single list
	return [item for sublist in lst for item in sublist]

def convert_to_binary_array_and_integer(A):
	"""
	Converts a list of integers into a binary array, where positive numbers are represented as 1 and non-positive numbers are represented as 0. Then, converts this binary array into an integer representation.
	
	Parameters:
	- A (list): A list of integers to be converted into a binary array and then into an integer.
	
	Returns:
	- int: The integer representation of the binary array formed from the list A, where each element in A contributes a bit to the binary number, with positive numbers as 1 and non-positive numbers as 0.
	"""
	# Convert list of integers into a binary array and then into an integer
	B = [(1 if x > 0 else 0) for x in A]
	integer_result = 0
	for bit in B:
		integer_result = (integer_result << 1) | bit
	return integer_result
	finalresult, active_workers = parallel_satisfy_ray(clause_sets, assignments)

def main(input_file_path, output_dir=None, output_path=None, selected_json_file=None, iterations=None, percentage=None, absolute=None, queue_size=None, default_size=False):
	"""
	The main entry point for the NDP-h. This function orchestrates the entire process of
	solving a SAT problem, from reading the input file, initializing the computing environment, to
	processing the problem and exporting the results.
	
	- It first determines the working directory and checks for the presence of an output directory.
	- It parses command-line arguments to get the input file path and an optional output directory.
	- It checks the existence of the input file and then reads its contents, extracting necessary
	  information like the DIMACS formula and variable assignments.
	- It initializes Ray for distributed computing and gathers cluster resources to determine the
	  computing capacity.
	- It generates a unique problem ID based on the input file and cluster resources, which is used
	  for naming output files.
	- It parses the DIMACS formula into a suitable data structure for processing.
	- It performs either Breadth-First only processing (BF) saving the results into a JSON or resumes
	  from BF or processes the complete input.
	- It launches parallel tasks to solve the SAT problem using Ray, distributing the workload across
	  available CPU cores.
	- It processes the results, attempting to derive RSA factors if applicable.
	- Finally, it exports the results to both text and CSV files, providing detailed information about
	  the problem, the computational process, and the outcomes.
	"""
	# Get the directory where the script is located
	script_directory = os.path.dirname(os.path.abspath(__file__))

	# Check if output_dir is provided, if not, use script's directory
	output_dir = output_dir if output_dir else script_directory
		
	# Function to get the total number of CPUs in the Ray cluster
	def get_total_gpus():
		cluster_resources = ray.cluster_resources()
		total_gpus = cluster_resources.get("GPU", 1)  # Defaults to 1 if not available
		return total_gpus
	
	# Function to get the total number of CPUs in the Ray cluster
	def get_total_cpus():
		cluster_resources = ray.cluster_resources()
		total_cpus = cluster_resources.get("CPU", 1)  # Defaults to 1 if not available
		return total_cpus

	if not os.path.exists(args.input_file_path):
		print(f"\nFile '{args.input_file_path}' not found.\n")
		return

	# Call read_input_files function to process input files
	dimacs_formula, input_digit, first_input_vars, second_input_vars, num_clauses, file_name, output_file_name_txt, output_file_name_csv = read_input_files(args.input_file_path, args.output_dir)

	# Initialize Ray and gather cluster resources
	cluster_resources, num_cpus, num_cpus, cluster_resources_info, memory_readable = initialize_and_print_cluster_resources()

	# Start timestamp and generate problem_id
	start_timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
	problem_data = {
		"Input File": file_name,
		"problem_type": "SAT",
	}

	# Generate problem ID including cluster resources and timestamps
	problem_id = generate_problem_id(problem_data, start_timestamp, cluster_resources)
		
	# Reduce problem ID to 5 digits for saving stats
	problem_id_short = problem_id[:5]
		
	# Get the base name of the current script
	script_name = os.path.basename(__file__)
	
	print(f"\nNDP started.")

	# Print the current UTC time stamp
	start_timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
	print("UTC", start_timestamp)
	print("Problem ID:", problem_id)

	# Set flag to False indicating not to proceed to satisfy parallel
	proceed_to_sp = False

	# Initialize clauseSets and assignments as empty lists
	clauseSets, assignments = [], []

	# Check if resuming from breadth-first
	if args.resume_from_bfs:

		# Get the total number of CPU cores available
		num_cores = get_total_cpus()

		try:
			# Load breadth-first search results and set flag to proceed to satisfy parallel
			clauseSets, assignments = load_bf_results(output_dir, num_cores)

			proceed_to_sp = True

		except FileNotFoundError:
			print("\nno BF results files found. Please run with -b first and ensure the BF results are saved.\n")
			sys.exit(1)

	# If proceeding to satisfy parallel mode and clauseSets and assignments are available
	if proceed_to_sp and clauseSets and assignments:
		futures = [satisfy_parallel.remote(clauseSet, assignment) for clauseSet, assignment in zip(clauseSets, assignments)]
		results = ray.get(futures)

	# Check if breadth-first only
	if args.breadth_first_only and not args.resume_from_bfs:
	
		# Check if assignments found
		S = parse_dimacs(dimacs_formula)
		
		# Extract additional information
		lines = dimacs_formula.strip().split('\n')
		_, _, num_vars, num_clauses = lines[0].split()	
		number_of_bits = second_input_vars[-1]  # Use the last integer in the list
		number_of_sbits = second_input_vars[0]  # Use the first integer in the list
		
		# Convert values to integer
		num_clauses = int(num_clauses)
		num_vars = int(num_vars)
		num_cores = int(get_total_cpus())

		# Check if no stopping criteria are provided as command-line arguments and prompt
		if args.queue_size is None and args.absolute is None and args.percentage is None and args.default_size is False:
			while True:
				"""Prompt user for inputs if not provided via CLI."""
				print("\nSelect an option for setting the queue or iterations:\n")
				print(f"     VARs: {num_vars}    CPUs: {num_cores}\n")
				print(" q:  set max queue size (power of 2)")
				print(" p:  set BF resolution percentage of VARs (0 - 99%)")
				print(f" a:  set absolute #VARs for BF resolution (0 < {num_vars} VARs)")
				print(f" d:  queue size next lower power of 2 of {num_cores} CPUs: queue size = {round_down_to_power_of_two(num_cores)}     (recommended)\n")
	
				user_choice = input("Your choice (q/p/a/d): ").strip()	
	
				if user_choice == "q":
				# Prompt for queue size
					queue_size_input = input("\nEnter max queue size (power of 2): ").strip()
					try:
						queue_size = int(queue_size_input)
						print(f"Queue size: {queue_size}\n")
						if queue_size > num_vars:
							print(f"Queue size cannot exceed {num_vars} VARs - retry.\n")
							continue
				
						elif not is_power_of_two(queue_size):  # Check if queue_size is NOT a power of two
							print(f"Adjust queue size to power of 2 - example: {round_down_to_power_of_two(num_cpus)}\n")
							continue
				
						else:
							args.queue_size = queue_size
							break
	
					except ValueError:
						print(f"\nQueue size must be positive number and to the power of 2 - example: {round_down_to_power_of_two(num_cpus)}\n")
						continue
	
				elif user_choice == "p":
					# Prompt for percentage
					percentage_input = input("\nEnter BF resolution percentage of VARs (0 - 99%): ").strip().replace('%', '')
					try:
						percentage = float(percentage_input)
						if 0 <= percentage <= 99.00:
							args.percentage = percentage
							print(f"\nPercentage of {num_vars} VARs set to: {percentage:.2f}%\n")
							break
						else:
							print("\nMax percentage value is 99% and must be positive - try again.\n")
							continue
					except ValueError:
						print("\nSomething went wrong - let's try it again.\n")
						continue
		
				elif user_choice == "a":
					# Prompt for absolute #VARs
					absolute_input = input(f"\nEnter absolute #VARs for BF resolution (0 < {num_vars}): ").strip()
					try:
						absolute = int(absolute_input)
						if absolute < num_vars:
							args.absolute = absolute
							print(f"\nAbsolute #VARs set to: {args.absolute}")
							break
						else:
							print("\nAbsolute number must be smaller than {num_vars}\n")
							continue
							
					except ValueError:
						print("\nInvalid input for absolute #VARs.")
						continue
			
				elif user_choice == "d":
					# Prompt for default queue size
					default_size = input(f"\nDefaulting to max queue size: {round_down_to_power_of_two(num_cores)}      hit enter to proceed.\n").strip()
					
					# Default to max queue size = num_cores rounded down to the next power of two
					args.queue_size = round_down_to_power_of_two(num_cores)
					break

		# Convert elapsed time to days, hours, minutes, and seconds
		def seconds_to_human_readable(seconds):
			components = []
			minutes, sec = divmod(seconds, 60)
			hours, minutes = divmod(minutes, 60)
			days, hours = divmod(hours, 24)
			months, days = divmod(days, 30)
			if months > 0:
				components.append(f"{int(months)} months")
			if days > 0:
				components.append(f"{int(days)} days")
			if hours > 0:
				components.append(f"{int(hours)} hours")
			if minutes > 0:
				components.append(f"{int(minutes)} minutes")
			components.append(f"{sec:.2f} seconds")
			return ", ".join(components)
			
		# Start time for breadth-first processing
		start_time_total = time.time()
		start_time_bfs = time.time()

		# Perform breadth-first search
		sizeOfCNF = num_vars 
		result, length, clauseSets, assignments, returned_factor, factor_source, returned_queue_size, iteration_count, default_size, percentage_stats_q, percentage_stats_a = satisfy_breadth_first_parallel_ray(S, sizeOfCNF, iterations, num_vars, percentage=args.percentage, absolute=args.absolute, queue_size=args.queue_size, default_size=args.default_size)

		# Calculate elapsed time for breadth-first processing
		end_time_bfs = time.time()
		bfs_processing_time = end_time_bfs - start_time_bfs
		
		# Define JSON output filenames
		output_file_name_json = os.path.join(str(output_dir), f"{os.path.splitext(file_name)[0]}_{problem_id_short}_{length}_q.json")

		# Set output path for BF results in script directory
		output_path = os.path.join(script_directory, output_file_name_json)
		
		# Save BF results to file
		save_bf_results(clauseSets, assignments, output_file_name_json)
		
		# Initialize a string to store the print statements
		print_statements = ""
	
		# Define output filenames including reduced problem ID and stopping size in percentage
		if args.percentage is not None:
			output_file_name_txt = os.path.join(str(output_dir), f"{os.path.splitext(file_name)[0]}_{problem_id_short}_{args.percentage}_p.txt")
			output_file_name_csv = os.path.join(str(output_dir), f"{os.path.splitext(file_name)[0]}_{problem_id_short}_{args.percentage}_p.csv")
		elif args.absolute is not None:
			output_file_name_txt = os.path.join(str(output_dir), f"{os.path.splitext(file_name)[0]}_{problem_id_short}_{returned_factor}_a.txt")
			output_file_name_csv = os.path.join(str(output_dir), f"{os.path.splitext(file_name)[0]}_{problem_id_short}_{returned_factor}_a.csv")
		elif args.queue_size is not None:
			output_file_name_txt = os.path.join(str(output_dir), f"{os.path.splitext(file_name)[0]}_{problem_id_short}_{length}_q.txt")
			output_file_name_csv = os.path.join(str(output_dir), f"{os.path.splitext(file_name)[0]}_{problem_id_short}_{length}_q.csv")
		else:
			output_file_name_txt = os.path.join(str(output_dir), f"{os.path.splitext(file_name)[0]}_{problem_id_short}_{length}_q.txt")
			output_file_name_csv = os.path.join(str(output_dir), f"{os.path.splitext(file_name)[0]}_{problem_id_short}_{length}_q.csv")
		
		# Format print statements for later export
		def format_for_export(seconds):
			human_readable = seconds_to_human_readable(seconds)
			return {
				"seconds": f"{seconds:.2f}",
				"human_readable": human_readable,
			}
		bfs_time_export = format_for_export(bfs_processing_time)
		export_data = {
			"Breadth-first Time (seconds)": bfs_time_export["seconds"],
			"Breadth-first Time (human-readable)": bfs_time_export["human_readable"],
		}

		# Store the print statements for later export
		print_statements += "     Input File: {}\n".format(file_name)
		print_statements += "           Bits: {}\n".format(number_of_bits)
		print_statements += "asymmetric Bits: {}\n".format(number_of_sbits)
		print_statements += "           VARs: {}\n".format(num_vars)
		print_statements += "        Clauses: {}\n".format(num_clauses)
		print_statements += "\n   Input Number: {}\n".format(input_digit)
		print_statements += "\n                   UTC start: {}\n".format(start_timestamp)
		print_statements += "                     UTC end: {}\n".format(time.strftime("%Y-%m-%d %H:%M:%S\n", time.gmtime()))

		# Print Stats		
		print(f"\n       +++ breath-first only +++\n\n")
		print("\n             code:", script_name)
		print(f'    NDP-h Version: {NDP_h_VERSION}\n')
		print("       Input File:", file_name)
		print(f"             Bits: {number_of_bits}")
		print(f"  asymmetric Bits: {number_of_sbits}\n")
		print(f"          Clauses: {num_clauses}")
		print(f"             VARs: {num_vars}")

		# Determine how to print the stopping condition based on the source
		if factor_source == "percentage":
			print(f"    Stopping size: {returned_factor} ({args.percentage:.2f}%)")
			print(f"       Queue size: {length}\n")
		elif factor_source == "absolute":
			print(f"    Stopping size: {returned_factor} ({percentage_stats_a:.2f}%)")
			print(f"       Queue size: {length}\n")
		elif factor_source == "queue_size":
			print(f"    Stopping size: {iteration_count} ({percentage_stats_q:.2f}%)")
			print(f"       Queue size: {len(result)}\n")
		else:
			print(f"    Stopping size: {iteration_count}\n")
		
		print(f"     Input Number: {input_digit}")
		print(f"\n          BF time: {bfs_time_export['human_readable']}")
		print(f"                   {bfs_time_export['seconds']} seconds\n")
		print("        UTC start:", start_timestamp)
		print("          UTC end:", time.strftime("%Y-%m-%d %H:%M:%S\n", time.gmtime()))
		print(f"\n       for Youcef.\n\n")
		
		# Write final queue size to text file
		with open(output_file_name_txt, 'w') as output_file_txt:
			output_file_txt.write(f"\n")
			output_file_txt.write(f"+++ breath-first only +++\n\n")
			output_file_txt.write(f"   code: {script_name}\n")
			output_file_txt.write(f"Version: {NDP_h_VERSION}\n\n")
			output_file_txt.write(f"   file: {file_name}\n")
			output_file_txt.write(f"     ID: {problem_id}\n\n")
			output_file_txt.write(f"           Bits: {number_of_bits}\n")
			output_file_txt.write(f"asymmetric Bits: {number_of_sbits}\n")
			output_file_txt.write(f"           VARs: {num_vars}\n")
			output_file_txt.write(f"        Clauses: {num_clauses}\n")
			output_file_txt.write(f"   Input Number: {input_digit}\n\n")
			output_file_txt.write(f"UTC start: {start_timestamp}\n")
			output_file_txt.write(f"  UTC end: {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}\n\n")
			output_file_txt.write(f"   BF time  (s): {bfs_time_export['seconds']}\n")
			output_file_txt.write(f"   BF time (HR): {bfs_time_export['human_readable']}\n\n")
			output_file_txt.write(f"        Clauses: {num_clauses}\n")
			output_file_txt.write(f"           VARs: {num_vars}\n")
			
			# Determine how to print the stopping condition based on the source
			if factor_source == "percentage":
				output_file_txt.write(f"  Stopping size: {returned_factor} ({args.percentage:.2f}%)\n")
				output_file_txt.write(f"     Queue size: {length}\n")
			elif factor_source == "absolute":
				output_file_txt.write(f"  Stopping size: {returned_factor} ({percentage_stats_a:.2f}%)\n")
				output_file_txt.write(f"     Queue size: {length}\n")
			elif factor_source == "queue_size":
				output_file_txt.write(f"  Stopping size: {iteration_count} ({percentage_stats_q:.2f}%)\n")
				output_file_txt.write(f"     Queue size: {length}\n")
			else:
				output_file_txt.write(f"  Stopping size: {iteration_count}\n")
	
		# Write final queue size to CSV file	
		with open(output_file_name_csv, 'w', newline='') as output_file_csv:
			csv_writer = csv.writer(output_file_csv)
			csv_writer.writerow(['Benchmark', 'Value'])
			csv_writer.writerow(["", ""])
			csv_writer.writerow(["", '+++ breath-first only +++'])
			csv_writer.writerow(["", ""])
			csv_writer.writerow(["code:", script_name])
			csv_writer.writerow(["Version:", NDP_h_VERSION])
			csv_writer.writerow(["file:",  file_name])
			csv_writer.writerow(["ID:",	 problem_id])
			csv_writer.writerow(["Bits:", number_of_bits])
			csv_writer.writerow(["asymmetric Bits:", number_of_sbits])
			csv_writer.writerow(["VARs:",  num_vars])
			csv_writer.writerow(["Clauses:",  num_clauses])
			csv_writer.writerow(["Input Number:",  input_digit])
			csv_writer.writerow(["UTC start:",	start_timestamp])
			csv_writer.writerow(["UTC end:",  time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())])
			csv_writer.writerow(["BF time (s):", bfs_time_export['seconds']])
			csv_writer.writerow(["BF time (HR):", bfs_time_export['human_readable']])
			csv_writer.writerow(["Clauses:",  num_clauses])
			csv_writer.writerow(["VARs:",  num_vars])
			
			# Determine how to print the stopping condition based on the source
			if factor_source == "percentage":
				csv_writer.writerow(["Stopping size:", returned_factor])
				csv_writer.writerow(["Stopping size (%):", f"{args.percentage:.2f}%"])
				csv_writer.writerow(["Queue size:", length])
			elif factor_source == "absolute":
				csv_writer.writerow(["Stopping size:", returned_factor])
				csv_writer.writerow(["Stopping size (%):", f"{percentage_stats_a:.2f}%"])
				csv_writer.writerow(["Queue size:", length])
			elif factor_source == "queue_size":
				csv_writer.writerow(["Stopping size:", iteration_count])
				csv_writer.writerow(["Stopping size (%):", f"{percentage_stats_q:.2f}%"])
				csv_writer.writerow(["Queue size:", length])
			else:
				csv_writer.writerow(["Stopping size:", iteration_count])
				csv_writer.writerow(["Stopping size (%):", f"{percentage_stats_q:.2f}%"])
				csv_writer.writerow(["Queue size:", length])

				
		print(f"  exporting stats to: {output_file_name_txt}")
		print(f"                      {output_file_name_csv}")
		print(f" BF results saved to: {output_path}\n\n")
		sys.exit()

	# Check if no breadth-first only and resume from breadth-first is set
	if not args.breadth_first_only and args.resume_from_bfs:

		# Check if assignments found
		S = parse_dimacs(dimacs_formula)

		# Extract additional information
		lines = dimacs_formula.strip().split('\n')
		_, _, num_vars, num_clauses = lines[0].split()
		number_of_bits = second_input_vars[-1]  # Use the last integer in the list
		number_of_sbits = second_input_vars[0]  # Use the first integer in the list

		# Convert values to integer
		num_clauses = int(num_clauses)

		# Convert elapsed time to days, hours, minutes, and seconds
		def seconds_to_human_readable(seconds):
			components = []
			minutes, sec = divmod(seconds, 60)
			hours, minutes = divmod(minutes, 60)
			days, hours = divmod(hours, 24)
			months, days = divmod(days, 30)
			if months > 0:
				components.append(f"{int(months)} months")
			if days > 0:
				components.append(f"{int(days)} days")
			if hours > 0:
				components.append(f"{int(hours)} hours")
			if minutes > 0:
				components.append(f"{int(minutes)} minutes")
			components.append(f"{sec:.2f} seconds")
			return ", ".join(components)

		# Start timing
		start_time_total = time.time()

		# Execute parallel processing and capture the results
		finalresult, active_workers = parallel_satisfy_ray(clauseSets, assignments)

		# Process the final result if necessary
		Assignment = non_zero_subarrays_recursive(finalresult)

		# Calculate timings
		end_time_total = time.time()
		total_processing_time = end_time_total - start_time_total

		# Format print statements for later export
		def format_for_export(seconds, total_processing_time):
			human_readable = seconds_to_human_readable(seconds)
			return {
				"seconds": f"{seconds:.2f}",
				"human_readable": human_readable,
			}
		total_time_export = format_for_export(total_processing_time, total_processing_time)

		# Define output filenames including problem_id
		output_file_name_txt = os.path.join(str(output_dir), f"{os.path.splitext(file_name)[0]}_{problem_id_short}.txt")
		output_file_name_csv = os.path.join(str(output_dir), f"{os.path.splitext(file_name)[0]}_{problem_id_short}.csv")

		# Initialize a string to store the print statements
		print_statements = ""

		# Initialize variables for RSA factors
		Li1 = [first_input_vars]
		Li2 = [second_input_vars]
		
		# Helper function to perform Miller-Rabin primality test
		def is_prime_miller_rabin(n, k=5):
			if n <= 1:
				return False
			if n <= 3:
				return True
			if n % 2 == 0:
				return False

			# Write n as 2^r * d + 1
			r, d = 0, n - 1
			while d % 2 == 0:
				r += 1
				d //= 2
				
			# Witness loop
			for _ in range(k):
				a = random.randint(2, n - 2)
				x = pow(a, d, n)
				if x == 1 or x == n - 1:
					continue
				for _ in range(r - 1):
					x = pow(x, 2, n)
					if x == n - 1:
						break
				else:
					return False  # n is definitely composite
					
			return True  # n is probably prime

		# Initialize rsa_fact1 and rsa_fact2 with default values
		rsa_fact1 = None
		rsa_fact2 = None

		# Store the print statements for later export
		print_statements += "     Input File: {}\n".format(file_name)
		print_statements += "           Bits: {}\n".format(number_of_bits)
		print_statements += "asymmetric Bits: {}\n".format(number_of_sbits)
		print_statements += "           VARs: {}\n".format(num_vars)
		print_statements += "        Clauses: {}\n".format(num_clauses)
		print_statements += "\n   Input Number: {}\n".format(input_digit)
		print_statements += " available CPUs: {}\n".format(int(num_cores))
		print_statements += "        workers: {}\n".format(int(active_workers))
		print_statements += "\n                   UTC start: {}\n".format(start_timestamp)
		print_statements += "                     UTC end: {}\n".format(time.strftime("%Y-%m-%d %H:%M:%S\n", time.gmtime()))
		print_statements += cluster_resources_info

		# Print Stats
		print(f"\n       +++ satisfy parallel only +++\n\n")
		print("\n             code:", script_name)
		print(f'    NDP-h Version: {NDP_h_VERSION}\n')
		print("       Input File:", file_name)
		print(f"             Bits: {number_of_bits}")
		print(f"  asymmetric Bits: {number_of_sbits}")
		print(f"             VARs: {num_vars}")
		print(f"          Clauses: {num_clauses}\n")
		print(f"     Input Number: {input_digit}")

		# Check if assignments found
		if (Assignment is not None):
			# Initialize Lo1 and Lo2 as empty lists or with default values
			Lo1, Lo2 = [], []

			# Ensure Assignment is processed to populate Lo1 and Lo2
			Lo1, Lo2 = replace_elements(Assignment, [first_input_vars], [second_input_vars])
			
			# Assuming Lo1 and Lo2 are now populated, flatten them
			Lo1_flat = flatten_list_of_lists(Lo1)
			Lo2_flat = flatten_list_of_lists(Lo2)

			# Then pass the flattened lists to the function
			rsa_fact1 = convert_to_binary_array_and_integer(Lo1_flat)
			rsa_fact2 = convert_to_binary_array_and_integer(Lo2_flat)

			# Check if the product of the factors equals the input number
			if rsa_fact1 * rsa_fact2 == input_digit:
			
				if is_prime_miller_rabin(rsa_fact1) and is_prime_miller_rabin(rsa_fact2):
					print(f"        RSA FACT1: {rsa_fact1}")
					print(f"        RSA FACT2: {rsa_fact2}")
					print(f"                   verified.\n")
				else:
					print(f"                   is not a product of two prime numbers (RSA) nor a prime number itself.\n")
			else:
				print(f"                   {input_digit} is prime!\n")

		print(f"   available CPUs: {int(num_cores)}")
		print("          workers:", active_workers)
		print("\n")

		# Format print statements for later export
		export_data = {
			"Total Time (seconds)": total_time_export["seconds"],
			"Total Time (human-readable)": total_time_export["human_readable"],
		}

		print(f"\n\n         NDP time: {total_time_export['human_readable']}")
		print(f"                   {total_time_export['seconds']} seconds")
		print("\n        UTC start:", start_timestamp)
		print("          UTC end:", time.strftime("%Y-%m-%d %H:%M:%S\n", time.gmtime()))
		print(f"\n       for Youcef.\n\n")

		# Convert the Assignment list to a string with line breaks after every 5 integers
		assignment_str = '\n'.join(', '.join(map(str, Assignment[i:i+13])) for i in range(0, len(Assignment), 13))

		try:
			# Write final queue size to text file
			with open(output_file_name_txt, 'w') as output_file_txt:
				output_file_txt.write(f"\n")
				output_file_txt.write(f"+++ satisfy parallel only +++\n\n")
				output_file_txt.write(f"   code: {script_name}\n")
				output_file_txt.write(f"Version: {NDP_h_VERSION}\n\n")
				output_file_txt.write(f"   file: {file_name}\n")
				output_file_txt.write(f"     ID: {problem_id}\n\n")
				output_file_txt.write(f"           Bits: {number_of_bits}\n")
				output_file_txt.write(f"asymmetric Bits: {number_of_sbits}\n")
				output_file_txt.write(f"           VARs: {num_vars}\n")
				output_file_txt.write(f"        Clauses: {num_clauses}\n")
				output_file_txt.write(f"   Input Number: {input_digit}\n\n")
				output_file_txt.write(f"UTC start: {start_timestamp}\n")
				output_file_txt.write(f"  UTC end: {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}\n\n")
				output_file_txt.write(f"Total time  (s): {total_time_export['seconds']}\n")
				output_file_txt.write(f"Total time (HR): {total_time_export['human_readable']}\n\n")
				output_file_txt.write(f"        Clauses: {num_clauses}\n")
				output_file_txt.write(f"           VARs: {num_vars}\n\n")
				output_file_txt.write(f" available CPUs: {int(num_cores)}\n")
				output_file_txt.write(f"        workers: {int(active_workers)}\n\n")
				output_file_txt.write(f"   Input Number: {input_digit}\n")

				# Check if the product of the factors equals the input number
				if rsa_fact1 * rsa_fact2 == input_digit:

				# Conditionally write RSA factors
					if is_prime_miller_rabin(rsa_fact1) and is_prime_miller_rabin(rsa_fact2):
						output_file_txt.write(f"      RSA FACT1: {rsa_fact1}\n")
						output_file_txt.write(f"      RSA FACT2: {rsa_fact2}\n")
						output_file_txt.write(f"                 verified.\n\n")
					else:
						output_file_txt.write("      RSA FACT1: none\n")
						output_file_txt.write("      RSA FACT2: none\n\n")
						output_file_txt.write(f"                 {input_digit} is not a product of two prime numbers (RSA) nor a prime number itself.\n\n")
				else:
					output_file_txt.write(f"                 {input_digit} is prime!\n\n")
				output_file_txt.write(f"\n\n{cluster_resources_info}\n")
				output_file_txt.write("\n")
				output_file_txt.write(f" Assignments: {Assignment}\n")

			print(f"  exporting stats to: {output_file_name_txt}")
			print(f"                      {output_file_name_csv}")
			
			# Write final queue size to CSV file
			with open(output_file_name_csv, 'w', newline='') as output_file_csv:
				csv_writer = csv.writer(output_file_csv)
				csv_writer.writerow(['Benchmark', 'Value'])
				csv_writer.writerow(["", ""])
				csv_writer.writerow(["", '+++ satisfy parallel only +++'])
				csv_writer.writerow(["", ""])
				csv_writer.writerow(["code:", script_name])
				csv_writer.writerow(["Version:", NDP_h_VERSION])
				csv_writer.writerow(["file:",  file_name])
				csv_writer.writerow(["ID:",	 problem_id])
				csv_writer.writerow(["Bits:", number_of_bits])
				csv_writer.writerow(["asymmetric Bits:", number_of_sbits])
				csv_writer.writerow(["VARs:",  num_vars])
				csv_writer.writerow(["Clauses:",  num_clauses])
				csv_writer.writerow(["Input Number:",  input_digit])
				csv_writer.writerow(["UTC start:",	start_timestamp])
				csv_writer.writerow(["UTC end:",  time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())])
				csv_writer.writerow(["Total time (seconds):", total_time_export['seconds']])
				csv_writer.writerow(["Total time (HR):", total_time_export['human_readable']])
				csv_writer.writerow(["Clauses:",  num_clauses])
				csv_writer.writerow(["VARs:",  num_vars])
				csv_writer.writerow(["available CPUs:",	 int(num_cores)])
				csv_writer.writerow(["workers:",	 int(active_workers)])
				csv_writer.writerow(["Input Number:", input_digit])
				
				# Check if the product of the factors equals the input number
				if rsa_fact1 * rsa_fact2 == input_digit:

					# Conditionally write RSA factors				
					if is_prime_miller_rabin(rsa_fact1) and is_prime_miller_rabin(rsa_fact2):
						csv_writer.writerow(["RSA FACT1:", rsa_fact1])
						csv_writer.writerow(["RSA FACT2:", rsa_fact2])
					else:
						csv_writer.writerow(["RSA FACT1:", "none"])
						csv_writer.writerow(["RSA FACT2:", "none"])
						csv_writer.writerow(["Note:", f"{input_digit} is not a product of two prime numbers (RSA) nor a prime number itself."])
				else:
					csv_writer.writerow(["RSA FACT1:", "none"])
					csv_writer.writerow(["RSA FACT2:", "none"])
					csv_writer.writerow(["Note:", f"{input_digit} is prime!"])
				csv_writer.writerow(["Cluster info:", cluster_resources_info])
				csv_writer.writerow(["Assignments:", assignment_str])

			print(f"done.\n")

		except Exception as e:
			print(f"\nAn error occurred while exporting - check export location for integrity/free space: {str(e)}\n\n")

	if not args.breadth_first_only and not args.resume_from_bfs:
	
		# Check if assignments found
		S = parse_dimacs(dimacs_formula)
		
		# Extract additional information
		lines = dimacs_formula.strip().split('\n')
		_, _, num_vars, num_clauses = lines[0].split()
		number_of_bits = second_input_vars[-1]  # Use the last integer in the list
		number_of_sbits = second_input_vars[0]  # Use the first integer in the list

		# Convert values to integer
		num_clauses = int(num_clauses)
		num_vars = int(num_vars)
		num_cores = int(get_total_cpus())

		# Check if no stopping criteria are provided as command-line arguments and prompt
		if args.queue_size is None and args.absolute is None and args.percentage is None and args.default_size is False:
			while True:
				"""Prompt user for inputs if not provided via CLI."""
				print("\nSelect an option for setting the queue or iterations:\n")
				print(f"     VARs: {num_vars}    CPUs: {num_cores}\n")
				print(" q:  set max queue size (power of 2)")
				print(" p:  set BF resolution percentage of VARs (0 - 99%)")
				print(f" a:  set absolute #VARs for BF resolution (0 < {num_vars} VARs)")
				print(f" d:  queue size next lower power of 2 of {num_cores} CPUs: queue size = {round_down_to_power_of_two(num_cores)}     (recommended)\n")
	
				user_choice = input("Your choice (q/p/a/d): ").strip()	
	
				if user_choice == "q":
				# Prompt for queue size
					queue_size_input = input("\nEnter max queue size (power of 2): ").strip()
					try:
						queue_size = int(queue_size_input)
						print(f"Queue size: {queue_size}\n")
						if queue_size > num_vars:
							print(f"Queue size cannot exceed {num_vars} VARs - retry.\n")
							continue
				
						elif not is_power_of_two(queue_size):  # Check if queue_size is NOT a power of two
							print(f"Adjust queue size to power of 2 - example: {round_down_to_power_of_two(num_cpus)}\n")
							continue
				
						else:
							args.queue_size = queue_size
							break
	
					except ValueError:
						print(f"\nQueue size must be positive number and to the power of 2 - example: {round_down_to_power_of_two(num_cpus)}\n")
						continue
	
				elif user_choice == "p":
					# Prompt for percentage
					percentage_input = input("\nEnter BF resolution percentage of VARs (0 - 99%): ").strip().replace('%', '')
					try:
						percentage = float(percentage_input)
						if 0 <= percentage <= 99.00:
							args.percentage = percentage
							print(f"\nPercentage of {num_vars} VARs set to: {percentage:.2f}%\n")
							break
						else:
							print("\nMax percentage value is 99% and must be positive - try again.\n")
							continue
					except ValueError:
						print("\nSomething went wrong - let's try it again.\n")
						continue
		
				elif user_choice == "a":
					# Prompt for absolute #VARs
					absolute_input = input(f"\nEnter absolute #VARs for BF resolution (0 < {num_vars}): ").strip()
					try:
						absolute = int(absolute_input)
						if absolute < num_vars:
							args.absolute = absolute
							print(f"\nAbsolute #VARs set to: {args.absolute}")
							break
						else:
							print("\nAbsolute number must be smaller than {num_vars}\n")
							continue
							
					except ValueError:
						print("\nInvalid input for absolute #VARs.")
						continue
			
				elif user_choice == "d":
					# Prompt for default queue size
					default_size = input(f"\nDefaulting to max queue size: {round_down_to_power_of_two(num_cores)}      hit enter to proceed.\n").strip()
					
					# Default to max queue size = num_cores rounded down to the next power of two
					args.queue_size = round_down_to_power_of_two(num_cores)
					break

		# Convert elapsed time to days, hours, minutes, and seconds
		def seconds_to_human_readable(seconds):
			components = []
			minutes, sec = divmod(seconds, 60)
			hours, minutes = divmod(minutes, 60)
			days, hours = divmod(hours, 24)
			months, days = divmod(days, 30)
			if months > 0:
				components.append(f"{int(months)} months")
			if days > 0:
				components.append(f"{int(days)} days")
			if hours > 0:
				components.append(f"{int(hours)} hours")
			if minutes > 0:
				components.append(f"{int(minutes)} minutes")
			components.append(f"{sec:.2f} seconds")
			return ", ".join(components)

		# Start total time and breadth-first processing time
		start_time_total = time.time()
		start_time_bfs = time.time()

		# Perform breadth-first search
		sizeOfCNF = num_vars 
		result, length, clauseSets, assignments, returned_factor, factor_source, returned_queue_size, iteration_count, default_size, percentage_stats_q, percentage_stats_a = satisfy_breadth_first_parallel_ray(S, sizeOfCNF, iterations, num_vars, percentage=args.percentage, absolute=args.absolute, queue_size=args.queue_size, default_size=args.default_size)

		# Calculate elapsed time for breadth-first processing
		end_time_bfs = time.time()
		bfs_processing_time = end_time_bfs - start_time_bfs
		print(f"\n              BF time: {bfs_processing_time:.2f} seconds\n")

		# If -s set save BF results to file
		if args.save:
			# Define JSON output filenames
			output_file_name_json = os.path.join(str(output_dir), f"{os.path.splitext(file_name)[0]}_{problem_id_short}_{length}_q.json")

			# Set output path for BF results in script directory
			output_path = os.path.join(script_directory, output_file_name_json)
		
			# Save BF results to file
			save_bf_results(clauseSets, assignments, output_file_name_json)

		
		print(f"\n\nevaluating {int(num_cores)} CPU cores for parallel processing..\n")

		# Execute parallel processing and capture the results
		finalresult, active_workers = parallel_satisfy_ray(clauseSets, assignments)

		#	Assignment = finalresult
		Assignment = non_zero_subarrays_recursive(finalresult)

		# Calculate timings
		end_time_total = time.time()
		total_processing_time = end_time_total - start_time_total
		parallel_processing_time = total_processing_time - bfs_processing_time
		parallel_percentage = (parallel_processing_time / total_processing_time) * 100
		bfs_percentage = (bfs_processing_time / total_processing_time) * 100

		# Format print statements for later export
		def format_for_export(seconds, total_processing_time, parallel_processing_time):
			
			human_readable = seconds_to_human_readable(seconds)
			return {
				"seconds": f"{seconds:.2f}",
				"human_readable": human_readable,
				"bfs_percentage": f"{bfs_percentage:.2f}",
				"parallel_percentage": f"{parallel_percentage:.2f}",
			}
		bfs_time_export = format_for_export(bfs_processing_time, total_processing_time, parallel_processing_time)
		bfs_percent_export = format_for_export(bfs_percentage, total_processing_time, parallel_processing_time)
		parallel_time_export = format_for_export(parallel_processing_time, total_processing_time, parallel_processing_time)
		parallel_percent_export = format_for_export(parallel_percentage, total_processing_time, parallel_processing_time)
		total_time_export = format_for_export(total_processing_time, total_processing_time, parallel_processing_time)

		# Initialize variables for RSA factors
		Li1 = [first_input_vars]
		Li2 = [second_input_vars]
		
		number_of_bits = second_input_vars[-1]  # Use the last integer in the list
		number_of_sbits = second_input_vars[0]  # Use the first integer in the list

		# Helper function to perform Miller-Rabin primality test
		def is_prime_miller_rabin(n, k=5):
			if n <= 1:
				return False
			if n <= 3:
				return True
			if n % 2 == 0:
				return False

			# Write n as 2^r * d + 1
			r, d = 0, n - 1
			while d % 2 == 0:
				r += 1
				d //= 2
				
			# Witness loop
			for _ in range(k):
				a = random.randint(2, n - 2)
				x = pow(a, d, n)
				if x == 1 or x == n - 1:
					continue
				for _ in range(r - 1):
					x = pow(x, 2, n)
					if x == n - 1:
						break
				else:
					return False  # n is definitely composite
					
			return True  # n is probably prime

		# Define output filenames including reduced problem ID and stopping size in percentage
		if args.percentage is not None:
			output_file_name_txt = os.path.join(str(output_dir), f"{os.path.splitext(file_name)[0]}_{problem_id_short}_{args.percentage}_p.txt")
			output_file_name_csv = os.path.join(str(output_dir), f"{os.path.splitext(file_name)[0]}_{problem_id_short}_{args.percentage}_p.csv")
		elif args.absolute is not None:
			output_file_name_txt = os.path.join(str(output_dir), f"{os.path.splitext(file_name)[0]}_{problem_id_short}_{returned_factor}_a.txt")
			output_file_name_csv = os.path.join(str(output_dir), f"{os.path.splitext(file_name)[0]}_{problem_id_short}_{returned_factor}_a.csv")
		elif args.queue_size is not None:
			output_file_name_txt = os.path.join(str(output_dir), f"{os.path.splitext(file_name)[0]}_{problem_id_short}_{length}_q.txt")
			output_file_name_csv = os.path.join(str(output_dir), f"{os.path.splitext(file_name)[0]}_{problem_id_short}_{length}_q.csv")
		else:
			output_file_name_txt = os.path.join(str(output_dir), f"{os.path.splitext(file_name)[0]}_{problem_id_short}_{length}_q.txt")
			output_file_name_csv = os.path.join(str(output_dir), f"{os.path.splitext(file_name)[0]}_{problem_id_short}_{length}_q.csv")

		# Initialize a string to store the print statements
		print_statements = ""

		# Initialize rsa_fact1 and rsa_fact2 with default values
		rsa_fact1 = None
		rsa_fact2 = None

		# Store the print statements for later export
		print_statements += "     Input File: {}\n".format(file_name)
		print_statements += "           Bits: {}\n".format(number_of_bits)
		print_statements += "asymmetric Bits: {}\n".format(number_of_sbits)
		print_statements += "           VARs: {}\n".format(num_vars)
		print_statements += "        Clauses: {}\n".format(num_clauses)
		print_statements += "\n   Input Number: {}\n".format(input_digit)
		print_statements += " available CPUs: {}\n".format(int(num_cores))
		print_statements += "        workers: {}\n".format(int(active_workers))
		print_statements += "\n                   UTC start: {}\n".format(start_timestamp)
		print_statements += "                     UTC end: {}\n".format(time.strftime("%Y-%m-%d %H:%M:%S\n", time.gmtime()))
		print_statements += cluster_resources_info

		# Print Stats
		print("\n             code:", script_name)
		print(f'    NDP-h Version: {NDP_h_VERSION}\n')
		print("       Input File:", file_name)
		print(f"             Bits: {number_of_bits}")
		print(f"  asymmetric Bits: {number_of_sbits}\n")
		print(f"          Clauses: {num_clauses}")
		print(f"             VARs: {num_vars}")
		
		# Determine how to print the stopping condition based on the source
		if factor_source == "percentage":
			print(f"    Stopping size: {returned_factor} ({args.percentage:.2f}%)")
			print(f"       Queue size: {length}\n")
		elif factor_source == "absolute":
			print(f"    Stopping size: {returned_factor} ({percentage_stats_a:.2f}%)")
			print(f"       Queue size: {length}\n")
		elif factor_source == "queue_size":
			print(f"    Stopping size: {iteration_count} ({percentage_stats_q:.2f}%)")
			print(f"       Queue size: {len(result)}\n")
		else:
			print(f"    Stopping size: {iteration_count}\n")
		
		print(f"     Input Number: {input_digit}")
		
		# Check if assignments found
		if (Assignment is not None):
		
			# Initialize Lo1 and Lo2 as empty lists or with default values
			Lo1, Lo2 = [], []

			# Ensure Assignment is processed to populate Lo1 and Lo2
			Lo1, Lo2 = replace_elements(Assignment, [first_input_vars], [second_input_vars])
			
			# Assuming Lo1 and Lo2 are now populated, flatten them
			Lo1_flat = flatten_list_of_lists(Lo1)
			Lo2_flat = flatten_list_of_lists(Lo2)

			# Then pass the flattened lists to the function
			rsa_fact1 = convert_to_binary_array_and_integer(Lo1_flat)
			rsa_fact2 = convert_to_binary_array_and_integer(Lo2_flat)
			
			# Check if the product of the factors equals the input number
			if rsa_fact1 * rsa_fact2 == input_digit:
			
				# Check if rsa_fact1 and rsa_fact2 are prime numbers using Miller-Rabin test
				if is_prime_miller_rabin(rsa_fact1) and is_prime_miller_rabin(rsa_fact2):
					print(f"        RSA FACT1: {rsa_fact1}")
					print(f"        RSA FACT2: {rsa_fact2}")
					print(f"                   verified.\n")
					
				else:
					print(f"                   is not a product of two prime numbers (RSA) nor a prime number itself.\n")

			else:
				# If no assignments found, it's prime
				print(f"                   {input_digit} is prime!\n")
				
		print(f"   available CPUs: {int(num_cores)}")
		print(f"       queue size: {len(result)}")
		print("          workers:", active_workers)
		print("\n")

		export_data = {
			"Breadth-first Time (seconds)": bfs_time_export["seconds"],
			"Breadth-first Time (human-readable)": bfs_time_export["human_readable"],
			"Breadth-first Time (bfs_percentage)": bfs_percent_export["bfs_percentage"],
			"Parallel Processing Time (seconds)": parallel_time_export["seconds"],
			"Parallel Processing Time (human-readable)": parallel_time_export["human_readable"],
			"Total Time (seconds)": total_time_export["seconds"],
			"Total Time (human-readable)": total_time_export["human_readable"],
		}
		print(f"Breadth-first processing time: {bfs_time_export['human_readable']}\n                               {bfs_time_export['seconds']} seconds\n                               {bfs_percentage:.2f} % ")
		print(f"     Parallel processing time: {parallel_time_export['human_readable']}\n                               {parallel_time_export['seconds']} seconds\n                               {parallel_percentage:.2f} % ")
		print(f"                     NDP time: {total_time_export['human_readable']}\n                               {total_time_export['seconds']} seconds\n")
		print("\n                    UTC start:", start_timestamp)
		print("                      UTC end:", time.strftime("%Y-%m-%d %H:%M:%S\n", time.gmtime()))
		print(f"\n                   for Youcef.\n\n")

		# Convert the Assignment list to a string with line breaks after every 5 integers
		assignment_str = '\n'.join(', '.join(map(str, Assignment[i:i+13])) for i in range(0, len(Assignment), 13))

		try:
			# Write final queue size to text file
			with open(output_file_name_txt, 'w') as output_file_txt:
				output_file_txt.write(f"   code: {script_name}\n")
				output_file_txt.write(f"Version: {NDP_h_VERSION}\n\n")
				output_file_txt.write(f"   file: {file_name}\n")
				output_file_txt.write(f"     ID: {problem_id}\n\n")
				output_file_txt.write(f"           Bits: {number_of_bits}\n")
				output_file_txt.write(f"asymmetric Bits: {number_of_sbits}\n")
				output_file_txt.write(f"           VARs: {num_vars}\n")
				output_file_txt.write(f"        Clauses: {num_clauses}\n")
				output_file_txt.write(f"   Input Number: {input_digit}\n\n")
				output_file_txt.write(f"UTC start: {start_timestamp}\n")
				output_file_txt.write(f"  UTC end: {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}\n\n")
				output_file_txt.write(f"Total time  (s): {total_time_export['seconds']}\n")
				output_file_txt.write(f"Total time (HR): {total_time_export['human_readable']}\n\n")
				output_file_txt.write(f"   BF time  (s): {bfs_time_export['seconds']}\n")
				output_file_txt.write(f"   BF time (HR): {bfs_time_export['human_readable']}\n")
				output_file_txt.write(f"   BF time  (%): {bfs_percent_export['bfs_percentage']}%\n\n")
				output_file_txt.write(f"        Clauses: {num_clauses}\n")
				output_file_txt.write(f"           VARs: {num_vars}\n")
				
				# Determine how to print the stopping condition based on the source
				if factor_source == "percentage":
					output_file_txt.write(f"  Stopping size: {returned_factor} ({args.percentage:.2f}%)\n")
					output_file_txt.write(f"     Queue size: {length}\n")
				elif factor_source == "absolute":
					output_file_txt.write(f"  Stopping size: {returned_factor} ({percentage_stats_a:.2f}%)\n")
					output_file_txt.write(f"     Queue size: {length}\n")
				elif factor_source == "queue_size":
					output_file_txt.write(f"  Stopping size: {iteration_count} ({percentage_stats_q:.2f}%)\n")
					output_file_txt.write(f"     Queue size: {length}\n")
				else:
					output_file_txt.write(f"  Stopping size: {iteration_count}\n")
			
				output_file_txt.write(f"\n   pp time  (s): {parallel_time_export['seconds']}\n")
				output_file_txt.write(f"   pp time (HR): {parallel_time_export['human_readable']}\n")
				output_file_txt.write(f"   pp time  (%): {parallel_percent_export['parallel_percentage']}%\n\n")
				output_file_txt.write(f" available CPUs: {int(num_cores)}\n")
				output_file_txt.write(f"        workers: {int(active_workers)}\n")
				output_file_txt.write(f"     queue size: {len(result)}\n\n")
				output_file_txt.write(f"   Input Number: {input_digit}\n")

				# Check if the product of the factors equals the input number
				if rsa_fact1 * rsa_fact2 == input_digit:

					# Conditionally write RSA factors
					if is_prime_miller_rabin(rsa_fact1) and is_prime_miller_rabin(rsa_fact2):
						output_file_txt.write(f"      RSA FACT1: {rsa_fact1}\n")
						output_file_txt.write(f"      RSA FACT2: {rsa_fact2}\n")
						output_file_txt.write(f"                 verified.\n\n")
					else:
						output_file_txt.write("      RSA FACT1: none\n")
						output_file_txt.write("      RSA FACT2: none\n\n")
						output_file_txt.write(f"                 {input_digit} is not a product of two prime numbers (RSA) nor a prime number itself.\n\n")
				else:
					output_file_txt.write(f"                 {input_digit} is prime!\n\n")
				output_file_txt.write(f"\n\n{cluster_resources_info}\n")
				output_file_txt.write("\n")
				output_file_txt.write(f" Assignments: {Assignment}\n")

			print(f"  exporting stats to: {output_file_name_txt}")
			print(f"                      {output_file_name_csv}")
			
			# If -s set print BF results saved to file
			if args.save:
				print(f"BF results saved to: {output_path}\n\n")

			# Write final queue size to CSV file
			with open(output_file_name_csv, 'w', newline='') as output_file_csv:
				csv_writer = csv.writer(output_file_csv)
				csv_writer.writerow(['Benchmark', 'Value'])
				csv_writer.writerow(["code:", script_name])
				csv_writer.writerow(["Version:", NDP_h_VERSION])
				csv_writer.writerow(["file:",  file_name])
				csv_writer.writerow(["ID:",	 problem_id])
				csv_writer.writerow(["Bits:", number_of_bits])
				csv_writer.writerow(["asymmetric Bits:", number_of_sbits])
				csv_writer.writerow(["VARs:",  num_vars])
				csv_writer.writerow(["Clauses:",  num_clauses])
				csv_writer.writerow(["Input Number:",  input_digit])
				csv_writer.writerow(["UTC start:",	start_timestamp])
				csv_writer.writerow(["UTC end:",  time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())])
				csv_writer.writerow(["Total time (seconds):", total_time_export['seconds']])
				csv_writer.writerow(["Total time (HR):", total_time_export['human_readable']])
				csv_writer.writerow(["BF time (s):", bfs_time_export['seconds']])
				csv_writer.writerow(["BF time (HR):", bfs_time_export['human_readable']])
				csv_writer.writerow(["BF time (%):", f"{bfs_percent_export['bfs_percentage']}%"])
				csv_writer.writerow(["Clauses:",  num_clauses])
				csv_writer.writerow(["VARs:",  num_vars])

				# Determine how to print the stopping condition based on the source
				if factor_source == "percentage":
					csv_writer.writerow(["Stopping size:", returned_factor])
					csv_writer.writerow(["Stopping size (%):", f"{args.percentage:.2f}%"])
					csv_writer.writerow(["Queue size:", length])
				elif factor_source == "absolute":
					csv_writer.writerow(["Stopping size:", returned_factor])
					csv_writer.writerow(["Stopping size (%):", f"{percentage_stats_a:.2f}%"])
					csv_writer.writerow(["Queue size:", length])
				elif factor_source == "queue_size":
					csv_writer.writerow(["Stopping size:", iteration_count])
					csv_writer.writerow(["Stopping size (%):", f"{percentage_stats_q:.2f}%"])
					csv_writer.writerow(["Queue size:", length])
				else:
					csv_writer.writerow(["Stop/Queue size:", iteration_count])
					
				csv_writer.writerow(["pp time (s):", parallel_time_export['seconds']])
				csv_writer.writerow(["pp time (HR):", parallel_time_export['human_readable']])
				csv_writer.writerow(["pp time (%):", f"{parallel_percent_export['parallel_percentage']}%"])
				csv_writer.writerow(["available CPUs:",	 int(num_cores)])
				csv_writer.writerow(["workers:",	 int(active_workers)])
				csv_writer.writerow(["Input Number:", input_digit])

				# Check if the product of the factors equals the input number
				if rsa_fact1 * rsa_fact2 == input_digit:

					# Conditionally write RSA factors
					if is_prime_miller_rabin(rsa_fact1) and is_prime_miller_rabin(rsa_fact2):
						csv_writer.writerow(["RSA FACT1:", rsa_fact1])
						csv_writer.writerow(["RSA FACT2:", rsa_fact2])
					else:
						csv_writer.writerow(["RSA FACT1:", "none"])
						csv_writer.writerow(["RSA FACT2:", "none"])
						csv_writer.writerow(["Note:", f"{input_digit} is not a product of two prime numbers (RSA) nor a prime number itself."])
				else:
					csv_writer.writerow(["RSA FACT1:", "none"])
					csv_writer.writerow(["RSA FACT2:", "none"])
					csv_writer.writerow(["Note:", f"{input_digit} is prime!"])
				csv_writer.writerow(["Cluster info:", cluster_resources_info])
				csv_writer.writerow(["Assignments:", assignment_str])

			print(f"done.\n")

		except Exception as e:
			print(f"\nAn error occurred while exporting - check export location for integrity/free space: {str(e)}\n\n")

if __name__ == "__main__":
	epilog_text = (
				"This command processes 'input_file.file', a DIMACS formatted file containing a factorization challenge, "
				"and saves the output files in the './outputs' directory.\n"
				"Ensure the input file is in the Paul Purdom and Amr Sabry's CNF Generator for Factoring Problems "
				"DIMACS CNF format for factorization problems.\n"
				"Visit https://cgi.luddy.indiana.edu/~sabry/cnf.html "
				"for details on generating such files or refer to the provided GitHub and IPFS resources for more "
				"information and source code.\n"
	)
	parser = argparse.ArgumentParser(
		description="Non-Deterministic Processor (NDP-h) - A Parallel SAT-Solver for Factorization Problems",
		epilog=epilog_text
	)
	parser.add_argument(
		"input_file_path",
		help="Path to the input file containing the Paul Purdom and Amr Sabry's CNF Generator for Factoring Problems DIMACS CNF formula and problem metadata."
	)
	parser.add_argument(
		"--output_dir",
		help="Optional: Specify the directory where output files will be saved. Defaults to the current directory if not provided.",
		default="."
	)
	parser.add_argument("-b", "--breadth_first_only", action="store_true", help="Terminate after breadth-first search and output results.")
	parser.add_argument("-r", "--resume_from_bfs", action="store_true", help="Resume from saved BF results and skip breadth-first search.")
	parser.add_argument("-p", "--percentage", type=float, help="Percentage of #VARs for CPU stopping size factor up to two decimals (0 - 99%).", default=None)
	parser.add_argument("-a", "--absolute", type=int, help="Absolute #VARs for CPU stopping size (0 < #VARs)", default=None)
	parser.add_argument("-q", "--queue_size", type=int, help="Max queue size (power of 2)")
	parser.add_argument("-d", "--default_size", action='store_true', help="Default to max queue size rounded up to the next lower power of 2 based on #CPUs")
	parser.add_argument('-s', '--save', action='store_true', help="Save BF results to the specified output file.")
	parser.add_argument('-v', '--version', action='version', version=f'NDP-h Version {NDP_h_VERSION}')

	args = parser.parse_args()
	
	if args.breadth_first_only and args.resume_from_bfs:
		print("\n-b (breadth-first only) and -r (resume from BF) cannot be used together. retry.\n")
		sys.exit(1)

	# Call the main function with the parsed arguments
	try:
		main(args.input_file_path, args.output_dir)
	except ValueError as e:
		# This will catch ValueError specifically and print just the message
		print(f"Error: {e}")
	except Exception as e:
		# This will catch any other general exception and prevent printing traceback
		print(f"An error occurred: {e}")