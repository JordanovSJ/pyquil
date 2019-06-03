from pyquil import Program, get_qc
from pyquil.gates import CNOT, CCNOT, H, MEASURE, RESET, RX, RY, RZ, X, Z, SWAP, CSWAP, I
import time
import matplotlib.pyplot as plt
import numpy as np
import sys
from typing import List
from qiskit.quantum_info import state_fidelity


def ghz_vector(n):

    vector = np.zeros(2**n)
    vector[0] = 1 / np.sqrt(2)
    vector[-1] = vector[0]
    return vector


def multi_CNOT(qubits: List[int], n) -> Program:
    program = Program()
    # create GHZ state
    program.inst(H(qubits[0]))
    program.inst(CNOT(qubits[0], qubits[1]))

    print(n)

    for i in range(n):
        program.inst(CNOT(qubits[0], qubits[1]))
        program.inst(I(qubits[0]))
        program.inst(I(qubits[1]))

        program.inst(CNOT(qubits[0], qubits[1]))
        program.inst(I(qubits[0]))
        program.inst(I(qubits[1]))

    return program


def multi_H(qubits: List[int], n) -> Program:
    program = Program()
    for i in range(n):
        program.inst(H(qubits[0]))
    return program


if __name__ == '__main__':

    simulation = int(sys.argv[1])
    simulation = (simulation == 0)
    gate = int(sys.argv[2])
    n = int(sys.argv[3])
    N = int(sys.argv[4])

    lattice = "Aspen-4-2Q-A"

    qpu = get_qc(lattice, as_qvm=simulation)
    qubits = qpu.device.qubits()

    print(f'All qubits on {lattice}: {qpu.device.qubits()}')
    print(f'\nSelected qubits: {qubits}')

    program_reset = Program(RESET())

    # program_reset.inst(H(qubits[0]))

    if gate == 0:
        print('CNOT')
        program_reset.inst(multi_CNOT(qubits, int(n/2)))
    elif gate == 1:
        print('H')
        program_reset.inst(multi_H(qubits, n))

    else:
        raise ValueError('invalid input args')

    # Measurements
    ro = program_reset.declare('ro', 'BIT', len(qubits))
    program_reset.inst([MEASURE(qubit, ro[idx]) for idx, qubit in enumerate(qubits)])

    program_reset.wrap_in_numshots_loop(N)
    binary_reset = qpu.compile(program_reset)

    start = time.time()
    results = qpu.run(binary_reset)
    total = time.time() - start
    print(f'\nExecution time with active reset: {total:.3f} s')

    # magic (there should be a less messy way to reformat those, but cant be bothered

    # reformat results
    results = list(results)
    for i, result in enumerate(results):
        results[i] = str(result).replace(',', '').replace(',', '').replace('[', '').replace(']', '').replace(' ', '')

    # get counts and the vector.
    counts = {}
    output_vector = np.zeros(4)
    for i in range(4):
        counts[str(bin(i)[2:]).zfill(2)] = 0
    for result in results:
        counts[result] += 1
        output_vector[int(result, 2)] += 1

    # normalize
    output_vector = output_vector/np.sqrt(sum(output_vector**2))
    fidelity = state_fidelity(output_vector, ghz_vector(2))

    # print(results)
    print(fidelity)
    print(counts)
    print(output_vector)

    plt.hist(results)
    plt.show()

    print('finito')
