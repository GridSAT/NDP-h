# NDP-h: Non-Deterministic Processor for SAT Solving.

## Introduction

The Non-Deterministic Processor (NDP-h) is a sophisticated parallel SAT-solver designed to efficiently solve CNFs tailored for Paul Purdom and Amr Sabry's Factoring Problems. This tool leverages advanced techniques in parallel processing and distributed computing to handle complex SAT challenges effectively.

## Features

- **Efficient Parallel SAT-solving:** Uses a parallel CPU breadth-first search to break down SAT problems into independently solvable sub-formulas.
- **Distributed Computing with Ray:** Employs the Ray framework to manage computational workload across multiple CPU cores efficiently.
- **Focus on Unit Clauses:** Prioritizes unit clauses to enhance the efficiency of the solving process.
- **First Satisfying Assignment:** Terminates upon finding the first satisfying assignment to maintain a low and almost constant memory footprint.
- **In honor of Youcef Hamadache.**

## Methodology

NDP-h initiates with a parallel breadth-first search to decompose the problem into smaller sub-formulas. Following this, it utilizes Ray for parallel processing, distributing tasks across a cluster to enhance performance. The solver is finely tuned to prioritize shorter clauses and focuses on unit clauses early in the process to streamline computation.

## Installation

### Prerequisites

- Python 3.x
- pip
- virtualenv (optional)
- Ray (for distributed computing)

### Steps

1. Start a screen session: `screen -S NDP-h`
2. Create and navigate to the project directory: `mkdir NDP-h && cd NDP-h`
3. Clone or copy the necessary files into the directory.
4. Install dependencies:

##### Prepare system virtual environment (virtualenv)

On linux run as root

```bash
sudo apt install python3-pip sysstat
```

##### Create virtual environment (virtualenv)

Log-in as user and run

```bash
cd /path/NDP-h

virtualenv pattern_solvers
```

### Activate and update virtual environment (virtualenv)

Login as user and run

```bash
cd /path/NDP-h

source NDP-h/bin/activate

pip install -r requirements
```

### Using Ray for Distributed Computing:

- [Follow Ray documentationor setup instructions](https://docs.ray.io/)
- Example commands for starting the head node (without ray dashboard) in a cluster with 4 worker nodes:

Start head node without Ray Dashboard - example initialization with 4 CPUs as system reserves:

```bash
export RAY_DISABLE_IMPORT_WARNING=1
CPUS=$(( $(lscpu --online --parse=CPU | egrep -v '^#' | wc -l) - 4 ))
ray start --head --include-dashboard=false --disable-usage-stats --num-cpus=$CPUS
```
		
Start worker nodes - example initialization with 1 CPUs system reserves:
```bash
export RAY_DISABLE_IMPORT_WARNING=1
CPUS=$(( $(lscpu --online --parse=CPU | egrep -v '^#' | wc -l) - 1 ))
ray start --address='MASTER-IP:6379' --redis-password='MASTER-PASSWORT' --num-cpus=$CPUS
```

### Run solver

example:

```bash
python3 NDP-h.py inputs/rsaFACT-64bit.dimacs -d
```


## Usage

Execute from the command line, specifying the path to the DIMACS formatted input file and optionally the output directory, e.g.:

CLI options: --input_file_path="inputs/INPUT.dimacs" --output_dir="outputs/"

### Options

#### Execution options

- -b Breath-First (BF) only outputting a result JSON
- -r Resume from BF only choosing the respective result JSON from a list
- -s Save BF JSON along the full NDP execution for later re-use

### BF-options

- -q define a Queue Size for BF
- -p any digit from 0 - 99% to specify % of VARs for BF
- -a absolute #VARs for BF
- -d default Queue Size setting next lower power of 2 of #CPUs (recommended)
				 
If BF-options are not specified via CLI you will be prompted to enter respective values

## Additional Resources

For generating DIMACS files or more information on the methodology, please visit:
- [Paul Purdom and Amr Sabry's CNF Generator](https://cgi.luddy.indiana.edu/~sabry/cnf.html)
- [GitHub - CNF Factoring Tool](https://github.com/GridSAT/CNF_FACT-MULT)
- IPFS - CNF Factoring Tool Source ipfs://QmYuzG46RnjhVXQj7sxficdRX2tUbzcTkSjZAKENMF5jba
