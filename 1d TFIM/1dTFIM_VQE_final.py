import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import IBMQ, Aer, BasicAer
from qiskit.compiler import transpile
from qiskit.quantum_info.operators import Operator, Pauli
from qiskit.quantum_info import process_fidelity
from qiskit.providers.aer import QasmSimulator
from qiskit.algorithms.optimizers import SPSA

backend = BasicAer.get_backend("statevector_simulator")

def initial_params(num_qubits, layers): #for now should return all zeroes
    thetas = np.zeros(layers * 2 * (num_qubits-1))
    return thetas

def loss(thetas, num_qubits, layers, shots, h_zz, h_z, h_x):
    initials = initial_state()
    circuit = ansatz(num_qubits, layers, thetas, initials)
    expected_value = get_EV(h_zz, h_z, h_x, circuit, shots)
    return expected_value

def initial_state(): #for now should return state of all zeroes
    initials = ''.join(['0' for i in range(num_qubits)])
    return initials

def ansatz(num_qubits, layers, thetas, initials):
    circuit = QuantumCircuit(num_qubits, num_qubits)
    circuit.initialize(initials, circuit.qubits)
    circuit.barrier()
    current_theta = 0
    for l in range(layers):
        for qubit in range(0, num_qubits-1, 2):
            circuit.cx(qubit, qubit+1)
            circuit.ry(thetas[current_theta], qubit+1, 'RY')
            current_theta += 1
        circuit.barrier()
        for qubit in range(1, num_qubits-1, 2):
            circuit.cx(qubit, qubit+1)
            circuit.ry(thetas[current_theta], qubit+1, 'RY')
            current_theta += 1
        circuit.barrier()
        for qubit in range(0, num_qubits-1, 2):
            circuit.cx(qubit+1, qubit)
            circuit.ry(thetas[current_theta], qubit, 'RY')
            current_theta += 1
        circuit.barrier()
        for qubit in range(1, num_qubits-1, 2):
            circuit.cx(qubit+1, qubit)
            circuit.ry(thetas[current_theta], qubit, 'RY')
            current_theta += 1
        circuit.barrier()
    return circuit

def evaluate(circuit, shots):
    circuit.measure_all() #different from circuit.measure([0,1], [0,1])?
    compiled = transpile(circuit, backend)
    job = backend.run(compiled, shots=shots)
    result = job.result()
    counts = result.get_counts(compiled)
    
    return counts

def bit_parity(substr):
    parity = 1
    for char in substr:
        if char == '1':
            parity *= -1
    return parity

def get_EV(h_zz, h_z, h_x, circuit, shots):
    counts = evaluate(circuit, shots)
    op_z = 0
    op_zz = 0
    for bitstr in counts:
        p = counts[bitstr]/shots #probability of getting bitstr
        for index in range(len(bitstr)-1, 0, -1): #single values
            substr = bitstr[index] #one number, 0 or 1
            parity = bit_parity(substr)
            op_z += p * parity
        for index in range(len(bitstr)-1, 1, -1): #goes to 1 to avoid going under array index 0
            substr = bitstr[index-1:index+1] #string of two numbers
            parity = bit_parity(substr)
            op_zz += p * parity
    
    for i in range(circuit.num_qubits):
        circuit.h(i)
    counts = evaluate(circuit, shots)
    op_x = 0
    for bitstr in counts:
        p = counts[bitstr]/shots
        for index in range(len(bitstr)-1, 0, -1):
            substr = bitstr[index]
            parity = bit_parity(substr)
            op_x += p * parity
    
    expected_value = h_zz * op_zz + h_z * op_z + h_x * op_x
    return expected_value

def optimize(maxiter, expected_value, initial_thetas): #expected value is loss function
    spsa = SPSA(maxiter=maxiter)
    solution = spsa.minimize(fun=expected_value, x0=initial_thetas)
    return solution



maxiter = 300
num_qubits = 5
layers = 2

def loss_final(thetas):
    num_qubits = 5
    layers = 2
    shots = 200
    h_zz = -1
    h_z = 1
    h_x = 1
    return loss(thetas, num_qubits, layers, shots, h_zz, h_z, h_x)



initial_thetas = initial_params(num_qubits, layers)
solution = optimize(maxiter, loss_final, initial_thetas)
print('Final thetas: ' + str(solution.x))
print('Final expected value (ground state energy): ' + str(solution.fun))