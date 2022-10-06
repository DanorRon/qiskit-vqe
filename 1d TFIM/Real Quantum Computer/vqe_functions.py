import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import IBMQ, Aer, BasicAer
from qiskit.compiler import transpile
from qiskit.quantum_info.operators import Operator, Pauli
from qiskit.quantum_info import process_fidelity
from qiskit.providers.aer import QasmSimulator
from qiskit.algorithms.optimizers import SPSA

class VQE:
    def __init__(self, num_qubits, layers, shots, h_zz, h_z, h_x, maxiter, backend):
        self.num_qubits = num_qubits
        self.layers = layers
        self.shots = shots
        self.h_zz = h_zz
        self.h_z = h_z
        self.h_x = h_x
        self.maxiter = maxiter
        self.backend = backend

    def initial_params(self): #for now should return all zeroes
        initial_thetas = np.zeros(self.layers * 2 * (self.num_qubits-1))
        return initial_thetas

    def ansatz(self, thetas):
        circuit = QuantumCircuit(self.num_qubits, self.num_qubits)
        #circuit.barrier()
        current_theta = 0
        for l in range(self.layers):
            for qubit in range(0, self.num_qubits-1, 2):
                circuit.cx(qubit, qubit+1)
                circuit.ry(thetas[current_theta], qubit+1, 'RY')
                current_theta += 1
            #circuit.barrier()
            for qubit in range(1, self.num_qubits-1, 2):
                circuit.cx(qubit, qubit+1)
                circuit.ry(thetas[current_theta], qubit+1, 'RY')
                current_theta += 1
            #circuit.barrier()
            for qubit in range(0, self.num_qubits-1, 2):
                circuit.cx(qubit+1, qubit)
                circuit.ry(thetas[current_theta], qubit, 'RY')
                current_theta += 1
            #circuit.barrier()
            for qubit in range(1, self.num_qubits-1, 2):
                circuit.cx(qubit+1, qubit)
                circuit.ry(thetas[current_theta], qubit, 'RY')
                current_theta += 1
            #circuit.barrier()
        return circuit

    def evaluate(self, circuit, backend):
        measured_circuit = circuit.measure_all(add_bits=False, inplace=False) #different from circuit.measure([0,1], [0,1])?
        compiled = transpile(measured_circuit, backend)
        job = backend.run(compiled, shots=self.shots)
        result = job.result()
        counts = result.get_counts(compiled)
        return counts

    def bit_parity(self, substr):
        parity = 1 - 2*(substr.count('1') % 2)
        return parity

    def get_EV(self, circuit, backend):
        counts = self.evaluate(circuit, backend)
        op_z = 0
        op_zz = 0
        num_qubits = circuit.num_qubits #number of qubits
        for bitstr in counts:
            p = counts[bitstr]/self.shots #probability of getting bitstr
            for index in range(num_qubits): #single values
                substr = bitstr[index] #one number, 0 or 1
                parity = self.bit_parity(substr)
                op_z += p * parity
            for index in range(num_qubits-1): #goes to 1 to avoid going under array index 0
                substr = bitstr[index:index+2] #string of two numbers
                parity = self.bit_parity(substr)
                op_zz += p * parity

        for i in range(num_qubits):
            circuit.h(i)
        counts = self.evaluate(circuit, backend)
        op_x = 0
        for bitstr in counts:
            p = counts[bitstr]/self.shots
            for index in range(num_qubits):
                substr = bitstr[index]
                parity = self.bit_parity(substr)
                op_x += p * parity
        
        expected_value = self.h_zz * op_zz + self.h_z * op_z + self.h_x * op_x
        return expected_value

    def optimize(self, loss, initial_thetas): #expected value is loss function
        spsa = SPSA(maxiter=self.maxiter)
        solution = spsa.minimize(fun=loss, x0=initial_thetas)
        return solution

    def loss(self, thetas):
        circuit = self.ansatz(thetas)
        loss = self.get_EV(circuit, self.backend)
        return loss