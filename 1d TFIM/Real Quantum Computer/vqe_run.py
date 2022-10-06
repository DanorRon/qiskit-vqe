import qiskit
from qiskit import Aer
import vqe_functions
from vqe_functions import VQE

backend = Aer.get_backend('aer_simulator')

num_qubits = 3
layers = 5
shots = 10000
h_zz = -1
h_z = 1
h_x = 1
maxiter = 300

vqe = VQE(num_qubits, layers, shots, h_zz, h_z, h_x, maxiter, backend)
initial_thetas = vqe.initial_params()
solution = vqe.optimize(vqe.loss, initial_thetas)
print('Final thetas: ' + str(solution.x))
print('Final expected value (ground state energy): ' + str(solution.fun))