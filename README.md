## TVMFuzz

A fuzzer for tensor-level intermediate representation(TIR) expressions in [TVM](https://tvm.apache.org/) created by David Pankratz for Secure Software Engineering Winter 2019 at the University of Alberta.

TIR is user-facing as it can be used in tasks such as implementing custom neural network operators or implementing existing operators using custom hardware instructions. TIR and its infrastructure is also used by TVM in the process of lowering from the Relay IR level to machine instruction level. Thus, fixing bugs in TIR is significant in improving the usability and correctness of TVM as a whole. 

## Organization

This section describes the layout of this repository:

- **bugs** contains bugs that were discovered by the TVMFuzz and fixed in TVM
- **docs** contains documentation of the design of TVMFuzz 
- **docker** contains dockerfile and local build instructions
- **settings** contains the settings for TVMFuzz 
- **src** contains the python source for TVMFuzz
- **quicktests** contains example of quicktest script for mismatches investigation

## Usage

To invoke the fuzzer simply run `python3 tvmfuzz.py`. This will generate a random `GenerationNode` tree which is capable of producing TVM and ground-truth programs.
To generate multiple instances use the `--R N` argument such as `python3 tvmfuzz.py --R 1000` to generate 1000 instances.

See this [article](https://github.com/dpankratz/CMPUT664Project/blob/master/docs/debugging.md) for more information about the output of the fuzzer.

## Ubuntu Installation

1. Follow the `automatic installation script` instructions for LLVM [here](https://apt.llvm.org/). For example LLVM 8 
2. Follow [TVM installation guide](https://docs.tvm.ai/install/from_source.html)
3. In the `config.cmake` file change `set(USE_LLVM OFF)` to `set(USE_LLVM /usr/bin/llvm-config-8)`
4. Clone this repository
5. Run `pip3 install --user termcolor`
6. Test the installation by running `python3 tvmfuzz.py`

If this is not amenable then a dockerfile and VM image are available.

## Dockerfile

To create locally:
1. `cd docker/`
2. `make build`
3. `docker run -it tvmfuzz`

Then the Installation can be tested via `python3 tvmfuzz.py`

To use remote image:
1. `docker pull dpankratz/tvmfuzz`
2. `docker run -it dpankratz/tvmfuzz`

## VM Image
Due to TVM periodically having breaking changes, TVMFuzz also includes a [Ubuntu VM image](https://drive.google.com/file/d/1b6WdQCPWoj9GQoVNGiw1kbwIMf0DDEIc/view?usp=sharing) that contains compatible versions of TVMFuzz, TVM, and LLVM.

To use this option:
1. install virtualbox
2. import VM
3. `cd tvmfuzz/src/`
4. `python3 tvmfuzz.py`


