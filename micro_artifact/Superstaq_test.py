from utils.resource_estimators import superstaq_estimate
from qiskit import QuantumCircuit, qasm2

benchmark_name = "IS-3D-30"
files = [f"t_circuits/{benchmark_name}_grid_c+t.qasm", f"t_circuits/{benchmark_name}_rustiq_c+t.qasm", f"t_circuits/{benchmark_name}_ncf_c+t.qasm"]
methods = ["grid", "rustiq", "ncf"]
print("With 1 T factories")
for file, method in zip(files, methods):
    qc = QuantumCircuit.from_qasm_file(file)
    metrics = superstaq_estimate(qc, num_t_factories=1)
    print(f"{method}: {metrics}")

print("--------------------------------")
print("With 10 T factories")
for file, method in zip(files, methods):
    qc = QuantumCircuit.from_qasm_file(file)
    metrics = superstaq_estimate(qc, num_t_factories=10)
    print(f"{method}: {metrics}")