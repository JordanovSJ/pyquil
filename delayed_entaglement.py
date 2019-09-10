from pyquil import Program, get_qc
from pyquil.gates import CNOT, CCNOT, H, MEASURE, RESET, RX, RY, RZ, X, Z, SWAP, CSWAP, I
import time
import matplotlib.pyplot as plt
import numpy as np
import sys
from typing import List
from qiskit.quantum_info import state_fidelity


def nCU1(axis, angle, ctrls, target) -> Program:
    """
    :param axis: {'x', 'y', 'z'}
    :param angle: REAL
    :param ctrls: list(int)
    :param target: int
    :return: Program
    """

    nCU1_program = Program()
    n_ctrls = len(ctrls)

    if n_ctrls == 1:
        if axis == 'x':
            nCU1_program.inst(RX(angle/2, target))
            nCU1_program.inst(CNOT(ctrls[0], target))
            nCU1_program.inst(RX(-angle/2, target))
            nCU1_program.inst(CNOT(ctrls[0], target))
        elif axis == 'y':
            nCU1_program.inst(RY(angle/2, target))
            nCU1_program.inst(CNOT(ctrls[0], target))
            nCU1_program.inst(RY(-angle/2, target))
            nCU1_program.inst(CNOT(ctrls[0], target))
        elif axis == 'z':
            nCU1_program.inst(RZ(angle/2, target))
            nCU1_program.inst(CNOT(ctrls[0], target))
            nCU1_program.inst(RZ(-angle/2, target))
            nCU1_program.inst(CNOT(ctrls[0], target))
        else:
            raise Exception(ValueError('Invalid axis value!'))
    elif n_ctrls > 1:
        nCU1_program.inst(nCU1(axis, angle/2, ctrls[1:], target))
        nCU1_program.inst(CNOT(ctrls[0], target))
        nCU1_program.inst(nCU1(axis, -angle/2, ctrls[1:], target))
        nCU1_program.inst(CNOT(ctrls[0], target))
    else:
        raise Exception(ValueError('Missing control qubits!'))

    return nCU1_program


# square root of SWAP http://tiny.cc/mues9y
def RSWAP(qubits)-> Program:
    i = qubits[0]
    j = qubits[1]
    RSWAP_program = Program()
    RSWAP_program.inst(CNOT(i, j))
    RSWAP_program.inst(H(i))
    RSWAP_program.inst(RZ(np.pi/4, i))
    RSWAP_program.inst(RZ(-np.pi/4, j))
    RSWAP_program.inst(CNOT(j, i))
    RSWAP_program.inst(RZ(-np.pi / 4, i))
    RSWAP_program.inst(H(i))
    RSWAP_program.inst(CNOT(i, j))
    RSWAP_program.inst(RZ(np.pi / 2, j))
    RSWAP_program.inst(RZ(-np.pi / 2, i))

    return RSWAP_program


# controlled RSWAP ( this is correct up to a phase?)
def cRSWAP(control, targets)-> Program:
    i = targets[0]
    j = targets[1]
    RSWAP_program = Program()

    RSWAP_program.inst(CCNOT(control, i, j))
    RSWAP_program.inst(H(i))
    RSWAP_program.inst(RZ(np.pi/4, i))
    RSWAP_program.inst(RZ(-np.pi/4, j))
    RSWAP_program.inst(CCNOT(control, j, i))
    RSWAP_program.inst(RZ(-np.pi / 4, i))
    RSWAP_program.inst(H(i))
    RSWAP_program.inst(CCNOT(control, i, j))

    RSWAP_program.inst(nCU1('z', np.pi / 2, [control], j))
    RSWAP_program.inst(nCU1('z', -np.pi / 2, [control], i))

    return RSWAP_program


if __name__ == '__main__':

    simulation = int(sys.argv[1])
    # simulation = 0
    simulation = (simulation == 0)
    N = 10000
    n_points = 50

    lattice = "Aspen-4-3Q-A"

    qpu = get_qc(lattice, as_qvm=simulation)
    qubits = qpu.device.qubits()

    print(f'All qubits on {lattice}: {qpu.device.qubits()}')
    print(f'\nSelected qubits: {qubits}')

    p101s = []
    p100s = []
    phis = np.arange(n_points)*2*np.pi/n_points
    for phi in phis:
        print(phi)

        program_reset = Program(RESET())

        ############### initiate state ###########################
        program_reset.inst(X(qubits[0]))
        program_reset.inst(H(qubits[2]))

        ###########################################################
        ############## Create the program #########################
        ###########################################################

        program_reset.inst(RSWAP(qubits[:2]))
        program_reset.inst(RZ(phi, qubits[0]))
        program_reset.inst(cRSWAP(qubits[2], qubits[:2]))

        # Measurements
        ro = program_reset.declare('ro', 'BIT', len(qubits))
        program_reset.inst([MEASURE(qubit, ro[idx]) for idx, qubit in enumerate(qubits)])

        program_reset.wrap_in_numshots_loop(N)
        binary_reset = qpu.compile(program_reset)

        results = qpu.run(binary_reset)

        # magic (there should be a less messy way to reformat those, but cant be bothered

        # reformat results
        results = list(results)
        for i, result in enumerate(results):
            results[i] = str(result).replace(',', '').replace(',', '').replace('[', '').replace(']', '').replace(' ', '')

        # get counts and the vector.
        counts = {}
        for i in range(2**len(qubits)):
            counts[str(bin(i)[2:]).zfill(len(qubits))] = 0
        for result in results:
            counts[result] += 1

        p100 = counts['100']/(counts['000']+counts['010']+counts['100']+counts['110'])
        p101 = counts['101']/(counts['001']+counts['011']+counts['101']+counts['111'])

        p100s.append(p100)
        p101s.append(p101)

    print(p100s)
    print(p101s)

    plt.plot(phis, p101s)
    plt.show()

    print('finito')