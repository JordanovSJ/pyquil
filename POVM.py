from pyquil import Program, get_qc
from pyquil.gates import CNOT, CCNOT, H, MEASURE, RESET, RX, RY, RZ, X, Z, SWAP, CSWAP
import time
import matplotlib.pyplot as plt
import numpy as np
import sys


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


def POVM_module_1(theta1, theta2, phi1, phi2, ancilla, target)-> Program:
    """
    :param theta1: REAL
    :param theta2: REAL
    :param phi1: REAL
    :param phi2: REAL
    :param ancilla: list(int)
    :param target: int
    :return: Program
    """

    POVM_module_1_program = Program()

    # BS1: split paths p1 and p2
    POVM_module_1_program.inst(CNOT(target, ancilla[0]))

    # Controlled rotation for path p1
    POVM_module_1_program.inst(X(ancilla[0]))
    POVM_module_1_program.inst(nCU1('y', 2*(theta1 + np.pi/2), [ancilla[0]], target))  # not sure if angle should not be 2x
    POVM_module_1_program.inst(X(ancilla[0]))

    # Controlled rotation for path p2
    if theta2 != 0:
        POVM_module_1_program.inst(nCU1('y', 2*theta2, [ancilla[0]], target))

    # Instead of branching to t-channel, perform swap of the ancilla qubit and the target
    POVM_module_1_program.inst(SWAP(target, ancilla[0]))

    # for the sake of clarity
    POVM_module_1_program.inst(X(ancilla[0]))
    POVM_module_1_program.inst(Z(ancilla[1]))

    # phase shifts
    if phi1 != 0:
        POVM_module_1_program.inst(X(ancilla[0]))
        POVM_module_1_program.inst(nCU1('z', phi1, [ancilla[0]], target))
        POVM_module_1_program.inst(X(ancilla[0]))

    if phi2 != 0:
        POVM_module_1_program.inst(nCU1('z', phi2, [ancilla[0]], target))

    return POVM_module_1_program


def POVM_module_2(theta3, theta4)-> Program:
    """
    :param theta3: REAL
    :param theta4: REAL
    :return: Program
    """

    POVM_module_2_program = Program()

    # bs2: split p3 and p4 channels
    POVM_module_2_program.inst(CCNOT(target, ancilla[0], ancilla[1]))

    # # p3 rotation
    POVM_module_2_program.inst(X(ancilla[1]))
    POVM_module_2_program.inst(nCU1('y', 2*(theta3 + np.pi/2), ancilla, target))
    POVM_module_2_program.inst(X(ancilla[1]))

    # p4 rotation
    if theta4 != 0:
        POVM_module_2_program.inst(nCU1('y', 2*theta4, ancilla, target))

    # SWAP
    POVM_module_2_program.inst(CSWAP(ancilla[0], target, ancilla[1]))
    POVM_module_2_program.inst(CNOT(ancilla[0], ancilla[1]))  # ??
    # POVM_module_2_program.inst(nCU1('z', np.pi, [ancilla[0]], ancilla[1]))  # fix a minus sign (not really necessary)

    return POVM_module_2_program


if __name__ == '__main__':

    # parameters
    simulation = (int(sys.argv[1]) == 1)
    POVM_parts = int(sys.argv[2])
    Kraus = int(sys.argv[3])
    shots = int(sys.argv[4])

    print('Simulation=' + str(simulation))
    print('POVM_parts='+str(POVM_parts))
    print('KRAUS ops='+str(Kraus))
    print('shots = ' + str(shots))

    if POVM_parts == 2:
        # initial state params
        theta0 = 2*np.pi/3
        phi0 = 0

        # module 1 parameters
        theta1 = np.pi/4
        theta2 = np.pi/4
        phi1 = 0
        phi2 = 0

    elif POVM_parts == 3:
        theta0 = 0
        phi0 = 0

        theta1 = np.arccos(np.sqrt(2/3))
        theta2 = np.pi/2
        theta3 = 0
        theta4 = np.pi/2
        phi1 = 0
        phi2 = 0

        alpha_ui = 0
        alpha_uii = -np.pi / 2

    else:
        raise Exception('Wrong arguments!')

    lattice = "Aspen-4-3Q-A"

    # for the purpose of POVM we choose the most connected qubit (10 in the case) to be the measured one
    qubits = [11, 10, 17]  # check if still correct
    ancilla = [11, 17]
    target = 10

    qpu = get_qc(lattice, as_qvm=simulation)

    print(f'All qubits on {lattice}: {qpu.device.qubits()}')
    print(f'\nSelected qubits: {qubits}')

    program = Program(RESET())
    # create initial state
    if theta0 != 0:
        program.inst(RY(theta0, target))
    if phi0 != 0:
        program.inst(RZ(phi0, target))

    # 1st AP module
    program.inst(POVM_module_1(theta1, theta2, phi1, phi2, ancilla, target))

    # 2nd AP module (+ U2, T1, T2, T3)
    if POVM_parts == 3:
        program.inst(nCU1('y', alpha_uii, [ancilla[0]], target))
        program.inst(POVM_module_2(theta3, theta4))

        # Kraus operators (Tsss)
        if Kraus == 1:
            # T2
            program.inst(X(ancilla[1]))
            program.inst(nCU1('y', 2*np.pi/3, ancilla, target))
            program.inst(X(ancilla[1]))

            # T3
            program.inst(nCU1('y', 7*np.pi/3, ancilla, target))

        # fix paths encodings: 11->01
        program.inst(CNOT(ancilla[1], ancilla[0]))  # adsfdsfdsds

    # Measurements
    ro = program.declare('ro', 'BIT', len(qubits))
    program.inst([MEASURE(qubit, ro[idx]) for idx, qubit in enumerate(qubits)])

    # No shots
    program.wrap_in_numshots_loop(shots)
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

    plt.hist(results)
    plt.show()
