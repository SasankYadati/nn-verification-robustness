from maraboupy import Marabou, MarabouUtils, MarabouCore
import numpy as np

# Load examples from the npz file
data = np.load('mnist_examples.npz')
x_0 = data['digit_0']
x_1 = data['digit_1']

delta = 0.001

onnx_filename = 'relu_dnn_mnist_classifier.onnx'

def verify_local_robustness(network_filename, x_input, true_label, delta_val):
    """
    Verifies delta-local robustness for a given input x.

    Checks if any other label can become the minimum score within the
    delta-neighborhood around x_input.

    Returns:
        'UNSAT' if robust (no counterexample found).
        'SAT' if not robust (counterexample exists).
        'Error' or other status if Marabou encounters issues.
    """
    print(f"\n--- Verifying robustness for label {true_label} with delta = {delta_val} ---")

    network = Marabou.read_onnx(network_filename)

    inputVars = network.inputVars[0].flatten()
    outputVars = network.outputVars[0].flatten()

    print(f"Input dimensions: {len(inputVars)}")
    print(f"Output dimensions: {len(outputVars)}")


    # 1. Set Input Bounds based on x_input and delta
    for i in range(len(inputVars)):
        lower_bound = max(0.0, x_input[i] - delta_val)
        upper_bound = min(1.0, x_input[i] + delta_val)
        network.setLowerBound(inputVars[i], lower_bound)
        network.setUpperBound(inputVars[i], upper_bound)

    # 2. Set Output Constraints (Looking for a counterexample)
    disjunction_constraints = []

    for j in range(len(outputVars)):
        if j == true_label:
            continue
        eq = MarabouUtils.Equation(MarabouCore.Equation.LE)
        eq.addAddend(-1.0, outputVars[j])
        eq.addAddend(1.0, outputVars[true_label])
        eq.setScalar(0.0)
        disjunction_constraints.append([eq])

    network.addDisjunctionConstraint(disjunction_constraints)

    # 3. Solve the Query
    print("Solving query...")
    options = Marabou.createOptions(verbosity = 1, timeoutInSeconds = 60)
    r1, vals, stats = network.solve(options=options)
    print(r1)


    # 4. Interpret Results
    if len(vals) > 0:
        print(
            f"Result: SAT (Found an adversarial example for label {true_label})"
        )
        return "SAT"
    elif stats.hasTimedOut():
        print(
             f"Result: TIMEOUT (Solver timed out for label {true_label})"
        )
        return "TIMEOUT"
    else:
        print(
            f"Result: UNSAT (Network is robust for label {true_label} with delta={delta_val})"
        )
        return "UNSAT"


result_0 = verify_local_robustness(onnx_filename, x_0, 0, delta)
result_1 = verify_local_robustness(onnx_filename, x_1, 1, delta)

print("\n--- Summary ---")
print(f"Robustness for digit 0 (delta={delta}): {result_0}")
print(f"Robustness for digit 1 (delta={delta}): {result_1}")
