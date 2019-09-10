from pyquil import Program, get_qc
from pyquil.gates import CNOT, H, MEASURE, RESET
from typing import List
import time
import matplotlib.pyplot as plt
import numpy as np
from qiskit.quantum_info import state_fidelity
import sys


def ghz_vector(n):

    vector = np.zeros(2**n)
    vector[0] = 1 / np.sqrt(2)
    vector[-1] = vector[0]
    return vector


def ghz_program(qubits: List[int]) -> Program:
    program = Program()
    program.inst(H(qubits[0]))
    for i in range(len(qubits) - 1):
        program.inst(CNOT(qubits[i], qubits[i + 1]))
    ro = program.declare('ro', 'BIT', len(qubits))
    program.inst([MEASURE(qubit, ro[idx]) for idx, qubit in enumerate(qubits)])
    return program


if __name__ == '__main__':

    n_qubits = int(sys.argv[1])
    lattice = "Aspen-4-16Q-A"
    device_qubits_ordered = [1,0,7,6,5,4,3,2,15,14,13,12,11,10,17,16]

    qpu = get_qc(lattice, as_qvm=False)
    qubits = device_qubits_ordered[:n_qubits]

    print(f'All qubits on {lattice}: {qpu.device.qubits()}')
    print(f'\nSelected qubits: {qubits}')

    program_reset = Program(RESET())
    program_reset.inst(ghz_program(qubits))
    program_reset.wrap_in_numshots_loop(10000)
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
    output_vector = np.zeros(2**n_qubits)
    for i in range(2**n_qubits):
        counts[str(bin(i)[2:]).zfill(n_qubits)] = 0
    for result in results:
        counts[result] += 1
        output_vector[int(result, 2)] += 1

    # normalize
    output_vector = output_vector/np.sqrt(sum(output_vector**2))
    fidelity = state_fidelity(output_vector, ghz_vector(n_qubits))

    # print(results)
    print(counts)
    print(output_vector)
    print(fidelity)

    # plt.hist(results)
    # plt.show()
