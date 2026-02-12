soft computing 

(practical - 1)Implementation of Fuzzy Logic Operations

import numpy as np

def fuzzy_union_or(A, B, operator='max'):
    
    if len(A) != len(B):
        raise ValueError("Fuzzy sets must have the same length (same universe of discourse).")
    
    if operator == 'max':
        
        return np.maximum(A, B)
    
    else:
        raise NotImplementedError(f"Operator '{operator}' not supported for Fuzzy Union.")

def fuzzy_intersection_and(A, B, operator='min'):
    
    if len(A) != len(B):
        raise ValueError("Fuzzy sets must have the same length (same universe of discourse).")
        
    if operator == 'min':
        
        return np.minimum(A, B)
  
    else:
        raise NotImplementedError(f"Operator '{operator}' not supported for Fuzzy Intersection.")

def fuzzy_complement_not(A):
    
    
    return 1 - A

U = np.array([1, 2, 3, 4, 5]) 
print(f"Universe of Discourse (U): {U}\n")


A = np.array([1.0, 0.8, 0.4, 0.1, 0.0])
B = np.array([0.0, 0.1, 0.3, 0.7, 1.0])

print("--- Original Sets ---")
print(f"Fuzzy Set A: {A}")
print(f"Fuzzy Set B: {B}\n")


# a. Fuzzy UNION (OR)
A_OR_B = fuzzy_union_or(A, B)
print("--- Fuzzy UNION (A OR B) ---")
print(f"Operation: max(mu_A(x), mu_B(x))")
# Example: max(A[0], B[0]) = max(1.0, 0.0) = 1.0
print(f"Result (A OR B): {A_OR_B}\n")

# b. Fuzzy INTERSECTION (AND)
A_AND_B = fuzzy_intersection_and(A, B)
print("--- Fuzzy INTERSECTION (A AND B) ---")
print(f"Operation: min(mu_A(x), mu_B(x))")
# Example: min(A[1], B[1]) = min(0.8, 0.1) = 0.1
print(f"Result (A AND B): {A_AND_B}\n")

# c. Fuzzy COMPLEMENT (NOT)
NOT_A = fuzzy_complement_not(A)
print("--- Fuzzy COMPLEMENT (NOT A) ---")
print(f"Operation: 1 - mu_A(x)")
# Example: 1 - A[2] = 1 - 0.4 = 0.6
print(f"Result (NOT A): {NOT_A}\n")

NOT_B = fuzzy_complement_not(B)
print("--- Fuzzy COMPLEMENT (NOT B) ---")
print(f"Operation: 1 - mu_B(x)")
print(f"Result (NOT B): {NOT_B}")


(practical - 2)Design of Fuzzy Inference System (FIS)

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# --- 1. Define the Antecedent (Input) and Consequent (Output) Variables ---

# New Antecedent/Consequent objects hold universe variables and membership functions
# The universe (range of possible values) for each variable is defined.
# Universe for 'service' quality, ranging from 0 to 10
service = ctrl.Antecedent(np.arange(0, 11, 1), 'service')
# Universe for 'food' quality, ranging from 0 to 10
food = ctrl.Antecedent(np.arange(0, 11, 1), 'food')
# Universe for 'tip' percentage, ranging from 0 to 25%
tip = ctrl.Consequent(np.arange(0, 26, 1), 'tip')

# --- 2. Define Fuzzy Membership Functions (Fuzzification) ---

# We use simple triangular and trapezoidal membership functions (trimf, trapmf)

# Service quality fuzzy sets
service['poor'] = fuzz.trimf(service.universe, [0, 0, 5])
service['acceptable'] = fuzz.trimf(service.universe, [0, 5, 10])
service['excellent'] = fuzz.trimf(service.universe, [5, 10, 10])

# Food quality fuzzy sets
food['bad'] = fuzz.trapmf(food.universe, [0, 0, 1, 3])
food['decent'] = fuzz.trimf(food.universe, [1, 5, 9])
food['great'] = fuzz.trapmf(food.universe, [7, 9, 10, 10])

# Tip percentage fuzzy sets
tip['low'] = fuzz.trimf(tip.universe, [0, 0, 13])
tip['medium'] = fuzz.trimf(tip.universe, [0, 13, 25])
tip['high'] = fuzz.trimf(tip.universe, [13, 25, 25])

# Optional: Visualize the membership functions
# service.view()
# food.view()
# tip.view()

# --- 3. Define the Fuzzy Rules (Rule Base) ---

# Rules are defined in a natural language-like IF-THEN format
rule1 = ctrl.Rule(service['poor'] | food['bad'], tip['low'])
rule2 = ctrl.Rule(service['acceptable'], tip['medium'])
rule3 = ctrl.Rule(service['excellent'] & food['great'], tip['high'])
rule4 = ctrl.Rule(food['decent'] & service['poor'], tip['medium'])

# --- 4. Create the Control System and Simulation ---

# The ControlSystem object holds the fuzzy rules
tip_control = ctrl.ControlSystem([rule1, rule2, rule3, rule4])

# The ControlSystemSimulation object allows us to pass inputs and get outputs
tipping_simulation = ctrl.ControlSystemSimulation(tip_control)

# --- 5. Fuzzify, Infer, Aggregate, and Defuzzify (The Process) ---

# Pass the input values to the simulation (Crisp Inputs)
# Example: Service is 6.5/10, Food is 9.8/10
tipping_simulation.input['service'] = 6.5
tipping_simulation.input['food'] = 9.8

# Compute the result (runs the entire FIS process: Fuzzification -> Inference -> Defuzzification)
tipping_simulation.compute()

# Get the crisp output value
tip_amount = tipping_simulation.output['tip']

# --- 6. Display Results ---

print(f"Service Rating: 6.5/10")
print(f"Food Rating: 9.8/10")
print(f"*** Recommended Tip: {tip_amount:.2f}% ***")

# Optional: Visualize the final result
# The `tip.view` method shows the aggregated output fuzzy set
# and the resulting crisp value (vertical line)
tip.view(sim=tipping_simulation)
plt.show()

# --- Example 2: Testing different inputs ---
print("\n--- Example 2: Poor Service/Bad Food ---")
tipping_simulation.input['service'] = 2
tipping_simulation.input['food'] = 3
tipping_simulation.compute()
print(f"Recommended Tip: {tipping_simulation.output['tip']:.2f}%")

(practical-3)Defuzzification Techniques

import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

def demonstrate_defuzzification(universe, aggregated_mf):
    """
    Applies and compares five common defuzzification techniques to a given 
    aggregated membership function (MF).
    """
    
    # 1. Centroid (CoG/CoM) - Most common and robust
    # Calculates the center of gravity of the area under the curve.
    cog = fuzz.defuzz(universe, aggregated_mf, 'centroid')
    
    # 2. Bisector (BoA)
    # Finds the vertical line that divides the area under the curve into two equal halves.
    boa = fuzz.defuzz(universe, aggregated_mf, 'bisector')
    
    # 3. Mean of Maximum (MoM)
    # Calculates the average of all points in the universe that have the maximum membership value (height).
    mom = fuzz.defuzz(universe, aggregated_mf, 'mom')
    
    # 4. Smallest of Maximum (SoM)
    # Calculates the smallest value in the universe that has the maximum membership value.
    som = fuzz.defuzz(universe, aggregated_mf, 'som')
    
    # 5. Largest of Maximum (LoM)
    # Calculates the largest value in the universe that has the maximum membership value.
    lom = fuzz.defuzz(universe, aggregated_mf, 'lom')
    
    # --- Print Results ---
    print("--- Defuzzification Results ---")
    print(f"Centroid (CoG):        {cog:.4f}")
    print(f"Bisector (BoA):        {boa:.4f}")
    print(f"Mean of Maximum (MoM): {mom:.4f}")
    print(f"Smallest of Max (SoM): {som:.4f}")
    print(f"Largest of Max (LoM):  {lom:.4f}")

    # --- Plot Visualization ---
    plt.figure(figsize=(10, 6))
    plt.plot(universe, aggregated_mf, 'b', linewidth=2.5, label='Aggregated Fuzzy Set')
    
    # Plot the results of each defuzzification method
    plt.axvline(cog, color='r', linestyle='--', label=f'Centroid ({cog:.2f})')
    plt.axvline(boa, color='g', linestyle='-.', label=f'Bisector ({boa:.2f})')
    plt.plot([mom, mom], [0, 1.0], 'k:', label=f'MoM ({mom:.2f})') # Using a vertical line for MoM
    plt.plot([som, som], [0, 1.0], 'c:', label=f'SoM ({som:.2f})')
    plt.plot([lom, lom], [0, 1.0], 'm:', label=f'LoM ({lom:.2f})')
    
    plt.title('Comparison of Defuzzification Techniques')
    plt.ylabel('Membership Degree')
    plt.xlabel('Output Universe')
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(True)
    plt.show()
    
# --- 1. Define the Universe of Discourse and Fuzzy Set ---

# The universe (range of possible output values, e.g., tip percentage 0 to 25)
X = np.arange(0, 26, 0.1)

# An example of an aggregated output fuzzy set (a trapezoid + a triangle)
# In a real FIS, this is the result of applying all rules (Inference) and 
# combining the outputs (Aggregation).
# We simulate a skewed/complex aggregated set to show differences in methods.
mf_1 = fuzz.trapmf(X, [0, 5, 8, 11])
mf_2 = fuzz.trimf(X, [9, 15, 25])

# The final aggregated set is usually the maximum (union) of all rule consequences
# Note: Since the output is the result of aggregation, it is an array of membership values.
aggregated_mf = np.fmax(mf_1 * 0.7, mf_2 * 1.0) # Assume mf_1 was scaled down by rule strength 0.7

# Run the demonstration
demonstrate_defuzzification(X, aggregated_mf)

(practical-4)Implementation of Single-Layer Perceptron

import numpy as np
import matplotlib.pyplot as plt

# --- 1. Define the Activation Function (Step Function) ---
def step_function(weighted_sum):
    """
    The Perceptron uses a Heaviside step function as its activation.
    It returns 1 (activate) if the weighted sum is non-negative, else 0 (deactivate).
    """
    return 1 if weighted_sum >= 0 else 0

# --- 2. Define the Perceptron Class ---
class Perceptron:
    """
    Implements the core learning logic of a Single-Layer Perceptron.
    """
    def __init__(self, num_inputs, learning_rate=0.1, max_epochs=100):
        # Initialize weights (W) and bias (b). We start with small random weights.
        # W has size (num_inputs) and b is a single scalar.
        self.weights = np.random.uniform(low=-0.5, high=0.5, size=num_inputs)
        self.bias = np.random.uniform(low=-0.5, high=0.5, size=1)
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        
        # History to track error over epochs
        self.errors = []

    def predict(self, inputs):
        """
        Calculates the weighted sum and applies the step function.
        Output = step( (W . X) + b )
        """
        # (W . X) is the dot product of weights and input features
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        
        # Apply the activation function
        return step_function(weighted_sum)

    def train(self, training_inputs, labels):
        """
        Trains the perceptron using the Perceptron Learning Rule.
        Weights are updated only when a misclassification occurs.
        """
        print(f"--- Training Perceptron (Epochs: {self.max_epochs}, Rate: {self.learning_rate}) ---")
        
        for epoch in range(self.max_epochs):
            total_error = 0
            
            # Iterate through each training example
            for inputs, label in zip(training_inputs, labels):
                
                # Forward Pass: Predict the output
                prediction = self.predict(inputs)
                
                # Calculate the error
                error = label - prediction
                total_error += abs(error)
                
                # Backward Pass: Update weights and bias only if error is non-zero
                if error != 0:
                    # Perceptron Update Rule: 
                    # W_new = W_old + LR * error * X
                    # b_new = b_old + LR * error * 1
                    self.weights += self.learning_rate * error * inputs
                    self.bias += self.learning_rate * error * 1 # Bias update
                    
            # Record error for tracking convergence
            self.errors.append(total_error)
            
            # Check for convergence: Stop if all samples are classified correctly
            if total_error == 0:
                print(f"Converged successfully at Epoch {epoch + 1}.")
                break
                
            if (epoch + 1) % 10 == 0:
                 print(f"Epoch {epoch + 1}/{self.max_epochs}, Total Error: {total_error}")

        print("\n--- Training Complete ---")
        print(f"Final Weights: {self.weights}")
        print(f"Final Bias: {self.bias[0]:.4f}")


# --- 3. Prepare Data for AND Gate (A Linearly Separable Problem) ---

# Input data (A, B)
X_train = np.array([
    [0, 0], # Input 1
    [0, 1], # Input 2
    [1, 0], # Input 3
    [1, 1]  # Input 4
])

# Output labels (A AND B)
y_train = np.array([0, 0, 0, 1])

# --- 4. Run the Perceptron Model ---

# Initialize the Perceptron with 2 inputs (features)
perceptron = Perceptron(num_inputs=X_train.shape[1], learning_rate=0.1, max_epochs=50)

# Train the model
perceptron.train(X_train, y_train)

# --- 5. Test the Trained Model ---

print("\n--- Testing Model Predictions ---")
test_cases = X_train
test_labels = y_train

for inputs, expected in zip(test_cases, test_labels):
    prediction = perceptron.predict(inputs)
    status = "Correct" if prediction == expected else "Incorrect"
    print(f"Input: {inputs}, Predicted: {prediction}, Expected: {expected} ({status})")

# Optional: Visualize the error history
plt.figure(figsize=(8, 4))
plt.plot(perceptron.errors, marker='o')
plt.title('Perceptron Training Error Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Total Misclassifications')
plt.grid(True)
plt.show()

(practical-5)Multilayer Perceptron using Back propagation Algorithm

import numpy as np

# --- 1. Define Activation and Derivative Functions (Sigmoid) ---

def sigmoid(x):
    """Sigmoid activation function: 1 / (1 + e^(-x))"""
    # Prevents overflow in e^(-x) for very large negative numbers
    x = np.clip(x, -500, 500) 
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(output):
    """Derivative of the sigmoid function, calculated using the output of the sigmoid."""
    # Derivative is O * (1 - O), where O is the output of the sigmoid.
    return output * (1 - output)

# --- 2. Define the Multilayer Perceptron Class ---
class MLP_Backpropagation:
    """
    Implements a simple two-layer Multilayer Perceptron (Input -> Hidden -> Output).
    """
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1, max_epochs=10000):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        
        # Initialize Weights and Biases randomly (W_ih: Input to Hidden, W_ho: Hidden to Output)
        # Weights should be initialized small to prevent saturation of the sigmoid function.
        self.W_ih = np.random.uniform(low=-0.5, high=0.5, size=(input_size, hidden_size))
        self.b_h = np.zeros((1, hidden_size)) # Bias for hidden layer
        
        self.W_ho = np.random.uniform(low=-0.5, high=0.5, size=(hidden_size, output_size))
        self.b_o = np.zeros((1, output_size)) # Bias for output layer
        
        self.errors = [] # To track mean squared error

    def forward_pass(self, X):
        """
        Calculates the output for a given input X.
        Z = WX + b; A = f(Z)
        """
        # Hidden Layer Calculation (Input -> Hidden)
        self.net_h = np.dot(X, self.W_ih) + self.b_h
        self.out_h = sigmoid(self.net_h)
        
        # Output Layer Calculation (Hidden -> Output)
        self.net_o = np.dot(self.out_h, self.W_ho) + self.b_o
        self.out_o = sigmoid(self.net_o)
        
        return self.out_o

    def backward_pass(self, X, y, out_o, out_h):
        """
        Calculates and applies weight updates based on the error.
        This is the core of the Backpropagation algorithm.
        """
        # --- A. Output Layer Error and Delta ---
        # Error = Target - Output
        error_o = y - out_o
        # Output Delta (d_o) = Error * f'(net_o)
        d_o = error_o * sigmoid_derivative(out_o)
        
        # --- B. Hidden Layer Error and Delta (Error Backpropagation) ---
        # Error_h = Delta_o * W_ho^T 
        # The error is propagated back using the weights W_ho.
        error_h = d_o.dot(self.W_ho.T)
        # Hidden Delta (d_h) = Error_h * f'(net_h)
        d_h = error_h * sigmoid_derivative(out_h)
        
        # --- C. Weight and Bias Updates (Gradient Descent) ---
        
        # Update W_ho (Hidden to Output Weights)
        # dW_ho = out_h.T . d_o
        self.W_ho += self.out_h.T.dot(d_o) * self.learning_rate
        self.b_o += np.sum(d_o, axis=0, keepdims=True) * self.learning_rate
        
        # Update W_ih (Input to Hidden Weights)
        # dW_ih = X.T . d_h
        self.W_ih += X.T.dot(d_h) * self.learning_rate
        self.b_h += np.sum(d_h, axis=0, keepdims=True) * self.learning_rate
        
        return np.mean(error_o**2)

    def train(self, X_train, y_train):
        """
        Main training loop.
        """
        print(f"--- Training MLP (Hidden Size: {self.hidden_size}, Rate: {self.learning_rate}) ---")
        
        for epoch in range(self.max_epochs):
            # 1. Forward Pass
            out_o = self.forward_pass(X_train)
            
            # 2. Backward Pass (Calculate Error and Update Weights)
            mse = self.backward_pass(X_train, y_train, out_o, self.out_h)
            
            self.errors.append(mse)
            
            if (epoch + 1) % 1000 == 0:
                print(f"Epoch {epoch + 1}/{self.max_epochs}, Mean Squared Error: {mse:.6f}")

        print("\n--- Training Complete ---")

# --- 3. Prepare Data for XOR Gate (A Non-Linearly Separable Problem) ---

# Input data (A, B) - 4 samples, 2 features
X_train = np.array([
    [0, 0], 
    [0, 1], 
    [1, 0], 
    [1, 1]  
])

# Output labels (A XOR B) - 4 samples, 1 output
y_train = np.array([
    [0], 
    [1], 
    [1], 
    [0]
])

# --- 4. Run the MLP Model ---

# Model Configuration: 2 inputs, 4 hidden neurons, 1 output
mlp = MLP_Backpropagation(
    input_size=2, 
    hidden_size=4, 
    output_size=1, 
    learning_rate=0.2, 
    max_epochs=10000
)

# Train the model
mlp.train(X_train, y_train)

# --- 5. Test the Trained Model ---

print("\n--- Testing Model Predictions ---")

# Pass the training data through the trained network
predictions = mlp.forward_pass(X_train)

for inputs, prediction, expected in zip(X_train, predictions, y_train):
    # Apply a threshold of 0.5 to convert the sigmoid output (0 to 1) into a binary prediction (0 or 1)
    predicted_class = 1 if prediction[0] >= 0.5 else 0
    status = "Correct" if predicted_class == expected[0] else "Incorrect"
    
    # Print prediction as a continuous value and its thresholded class
    print(f"Input: {inputs}, Output: {prediction[0]:.4f}, Predicted Class: {predicted_class}, Expected: {expected[0]} ({status})")

# Optional: Visualize the error history
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.plot(mlp.errors)
plt.title('MLP Training Error (Mean Squared Error) Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.grid(True)
plt.show()

(practical-6)Implementation of Simple Neural Network (McCulloh Pitts model)

Implementation of Simple Neural Network (McCulloh Pitts model)
import numpy as np

# --- 1. Activation Function (Threshold Logic) ---

def mcp_activation(net_input, threshold):
    """
    The MCP neuron uses a fixed threshold activation.
    Output = 1 (Fires) if net_input >= threshold, else 0 (Does not fire).
    """
    return 1 if net_input >= threshold else 0

# --- 2. The MCP Neuron Function ---

def mcp_neuron(inputs, weights, threshold):
    """
    Simulates the McCulloch-Pitts neuron calculation.
    net_input = Sum(input_i * weight_i)
    Output = mcp_activation(net_input, threshold)
    """
    # Convert inputs and weights to NumPy arrays for easy dot product
    inputs = np.array(inputs)
    weights = np.array(weights)
    
    # Calculate the net input (Weighted Sum)
    net_input = np.dot(inputs, weights)
    
    # Determine the output based on the fixed threshold
    output = mcp_activation(net_input, threshold)
    
    return net_input, output

# --- 3. Implementation of Logical Gates ---

def implement_or_gate():
    """Simulates the Logical OR gate (Output is 1 if EITHER input is 1)."""
    
    print("\n--- Implementing Logical OR Gate ---")
    
    # Fixed parameters for OR Gate:
    # We want (1*1) + (1*0) >= 1  -> 1 >= 1 (True)
    # We want (0*0) + (0*0) >= 1  -> 0 >= 1 (False)
    weights = [1, 1]  # Equal importance for both inputs
    threshold = 1     # Fire if at least one weighted input is 1
    
    test_cases = [
        ([0, 0], 0),
        ([0, 1], 1),
        ([1, 0], 1),
        ([1, 1], 1)
    ]
    
    print(f"Weights: {weights}, Threshold: {threshold}")
    print("Input (A, B) | Net Input | Output | Expected")
    print("-" * 38)
    
    for inputs, expected in test_cases:
        net_input, output = mcp_neuron(inputs, weights, threshold)
        print(f"  {inputs[0]}, {inputs[1]}    |   {net_input}      |    {output}   |   {expected}")


def implement_not_gate():
    """Simulates the Logical NOT gate (Output is the inverse of the input)."""
    
    print("\n--- Implementing Logical NOT Gate ---")
    
    # Fixed parameters for NOT Gate (Single Input):
    # We need a strong negative weight to inhibit firing when the input is 1.
    weights = [-1]    # Negative weight for the single input
    threshold = 0     # Needs to be low (non-positive)
    
    test_cases = [
        ([0], 1),
        ([1], 0)
    ]
    
    print(f"Weights: {weights}, Threshold: {threshold}")
    print("Input (A) | Net Input | Output | Expected")
    print("-" * 38)
    
    for inputs, expected in test_cases:
        net_input, output = mcp_neuron(inputs, weights, threshold)
        print(f"    {inputs[0]}   |    {net_input}      |    {output}   |   {expected}")

# --- 4. Run the Implementations ---

implement_or_gate()
implement_not_gate()




























