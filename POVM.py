from pyquil import Program, get_qc
from pyquil.gates import CNOT, H, MEASURE, RESET, RX, RY, RZ
from typing import List
import time
import matplotlib.pyplot as plt
import numpy as np
import sys


def nCU1(axis, angle, ctrls, target) -> Program:

    program = Program()
    n_ctrls = len(ctrls)

    if n_ctrls == 1:
        if axis == 'x':
            program.inst(RX(angle/2, target))
            program.inst(CNOT(ctrls[0], target))
            program.inst(RX(-angle/2, target))
            program.inst(CNOT(ctrls[0], target))
        elif axis == 'y':
            program.inst(RY(angle/2, target))
            program.inst(CNOT(ctrls[0], target))
            program.inst(RY(-angle/2, target))
            program.inst(CNOT(ctrls[0], target))
        elif axis == 'z':
            program.inst(RZ(angle/2, target))
            program.inst(CNOT(ctrls[0], target))
            program.inst(RZ(-angle/2, target))
            program.inst(CNOT(ctrls[0], target))
        else:
            raise Exception(ValueError('Invalid axis value!'))
    elif n_ctrls > 1:
        program.inst(nCU1(axis, angle/2, ctrls[1:], target))
        program.inst(CNOT(ctrls[0], target))
        program.inst(nCU1(axis, -angle/2, ctrls[1:], target))
        program.inst(CNOT(ctrls[0], target))
    else:
        raise Exception(ValueError('Missing control qubits!'))

    return program


if __name__ == '__main__':

    lattice = "Aspen-4-3Q-A"
    qubits = [17, 10, 11]  # check if still correct

    qpu = get_qc(lattice, as_qvm=True)

    print(f'All qubits on {lattice}: {qpu.device.qubits()}')
    print(f'\nSelected qubits: {qubits}')

    program = Program(RESET())
    program.inst(H(qubits[0]))
    program.inst(H(qubits[1]))
    program.inst(nCU1('y', np.pi/3, qubits[:2], qubits[-1]))
    # Measurements
    ro = program.declare('ro', 'BIT', len(qubits))
    program.inst([MEASURE(qubit, ro[idx]) for idx, qubit in enumerate(qubits)])

    # No shots
    program.wrap_in_numshots_loop(1000)
    # Compile to QPU
    binary_reset = qpu.compile(program)

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
    output_vector = np.zeros(8)
    for i in range(8):
        counts[str(bin(i)[2:]).zfill(3)] = 0
    for result in results:
        counts[result] += 1
        output_vector[int(result, 2)] += 1

    # normalize
    output_vector = output_vector/np.sqrt(sum(output_vector**2))

    # print(results)
    print(counts)
    print(output_vector)

    # plt.hist(results)
    # plt.show()