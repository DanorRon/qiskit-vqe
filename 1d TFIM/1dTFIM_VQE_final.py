import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import IBMQ, Aer, BasicAer
from qiskit.compiler import transpile
from qiskit.quantum_info.operators import Operator, Pauli
from qiskit.quantum_info import process_fidelity
from qiskit.providers.aer import QasmSimulator
from qiskit.algorithms.optimizers import SPSA

backend = Aer.get_backend('aer_simulator')

def initial_params(num_qubits, layers): #for now should return all zeroes
    thetas = np.zeros(layers * 2 * (num_qubits-1))
    return thetas

def ansatz(num_qubits, layers, thetas):
    circuit = QuantumCircuit(num_qubits, num_qubits)
    #circuit.barrier()
    current_theta = 0
    for l in range(layers):
        for qubit in range(0, num_qubits-1, 2):
            circuit.cx(qubit, qubit+1)
            circuit.ry(thetas[current_theta], qubit+1, 'RY')
            current_theta += 1
        #circuit.barrier()
        for qubit in range(1, num_qubits-1, 2):
            circuit.cx(qubit, qubit+1)
            circuit.ry(thetas[current_theta], qubit+1, 'RY')
            current_theta += 1
        #circuit.barrier()
        for qubit in range(0, num_qubits-1, 2):
            circuit.cx(qubit+1, qubit)
            circuit.ry(thetas[current_theta], qubit, 'RY')
            current_theta += 1
        #circuit.barrier()
        for qubit in range(1, num_qubits-1, 2):
            circuit.cx(qubit+1, qubit)
            circuit.ry(thetas[current_theta], qubit, 'RY')
            current_theta += 1
        #circuit.barrier()
    return circuit

def evaluate(circuit, shots):
    measured_circuit = circuit.measure_all(add_bits=False, inplace=False) #different from circuit.measure([0,1], [0,1])?
    compiled = transpile(measured_circuit, backend)
    job = backend.run(compiled, shots=shots)
    result = job.result()
    counts = result.get_counts(compiled)
    return counts

def bit_parity(substr):
    parity = 1 - 2*(substr.count('1') % 2)
    return parity

def get_EV(h_zz, h_z, h_x, circuit, shots):
    counts = evaluate(circuit, shots)
    op_z = 0
    op_zz = 0
    num_qubits = circuit.num_qubits #number of qubits
    for bitstr in counts:
        p = counts[bitstr]/shots #probability of getting bitstr
        for index in range(num_qubits): #single values
            substr = bitstr[index] #one number, 0 or 1
            parity = bit_parity(substr)
            op_z += p * parity
        for index in range(num_qubits-1): #goes to 1 to avoid going under array index 0
            substr = bitstr[index:index+2] #string of two numbers
            parity = bit_parity(substr)
            op_zz += p * parity

    for i in range(num_qubits):
        circuit.h(i)
    counts = evaluate(circuit, shots)
    op_x = 0
    for bitstr in counts:
        p = counts[bitstr]/shots
        for index in range(num_qubits):
            substr = bitstr[index]
            parity = bit_parity(substr)
            op_x += p * parity
    
    expected_value = h_zz * op_zz + h_z * op_z + h_x * op_x
    return expected_value

def optimize(maxiter, expected_value, initial_thetas): #expected value is loss function
    spsa = SPSA(maxiter=maxiter)
    solution = spsa.minimize(fun=expected_value, x0=initial_thetas)
    return solution

def loss(thetas, num_qubits, layers, shots, h_zz, h_z, h_x):
    circuit = ansatz(num_qubits, layers, thetas)
    expected_value = get_EV(h_zz, h_z, h_x, circuit, shots)
    return expected_value

maxiter = 300
num_qubits = 5
layers = 5

def loss_final(thetas):
    num_qubits = 3
    layers = 5 #TODO
    shots = 10000 #TODO
    h_zz = -1
    h_z = 1
    h_x = 1
    return loss(thetas, num_qubits, layers, shots, h_zz, h_z, h_x)

#circuit = QuantumCircuit(5,5)
#print(evaluate(circuit, 10000))
#print(get_EV(0, 0, 1, circuit, 10000))

initial_thetas = initial_params(num_qubits, layers)
solution = optimize(maxiter, loss_final, initial_thetas)
print('Final thetas: ' + str(solution.x))
print('Final expected value (ground state energy): ' + str(solution.fun))