# Exercise 1 - Draw lines y = 2x + 1, y = 2x + 2, y = 2x + 3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Prepare x values
x = np.linspace(-10, 10, 100)

# Plot the three lines
plt.figure(figsize=(8, 6))
plt.plot(x, 2 * x + 1, 'r-', label='y = 2x + 1')  # Red solid line
plt.plot(x, 2 * x + 2, 'g--', label='y = 2x + 2')  # Green dashed line
plt.plot(x, 2 * x + 3, 'b:', label='y = 2x + 3')   # Blue dotted line

# Add title and axis labels
plt.title('Lines with Slope 2 and Different Intercepts')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.legend()
plt.grid(True)
plt.show()


# Exercise 2 - Scatter plot of points (x, y)
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
y = np.array([-0.57, -2.57, -4.80, -7.36, -8.78, -10.52, -12.85, -14.69, -16.78])

plt.figure(figsize=(6, 5))
plt.plot(x, y, '+', markersize=10, color='black')
plt.title('Scatter Plot of Points')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()


# Exercise 3 - Read CSV and process height/weight data
# Make sure 'weight-height.csv' is in the same directory as your script
data = pd.read_csv("weight-height.csv")

# Extract height and weight
length = data['Height'].values  # in inches
weight = data['Weight'].values  # in pounds

# Convert to metric units
length_cm = length * 2.54
weight_kg = weight * 0.453592

# Calculate means
mean_length = np.mean(length_cm)
mean_weight = np.mean(weight_kg)
print(f"Mean Length (cm): {mean_length:.2f}")
print(f"Mean Weight (kg): {mean_weight:.2f}")

# Draw histogram of lengths
plt.figure(figsize=(7, 5))
plt.hist(length_cm, bins=20, color='skyblue', edgecolor='black')
plt.title("Histogram of Student Heights (cm)")
plt.xlabel("Height (cm)")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()


# Exercise 4 - Matrix inverse and identity check
A = np.array([[1, 2, 3],
              [0, 1, 4],
              [5, 6, 0]])

# Calculate inverse
A_inv = np.linalg.inv(A)

# Verify by multiplying with original matrix
identity1 = np.dot(A, A_inv)
identity2 = np.dot(A_inv, A)

print("A * A_inv:")
print(identity1)
print("\nA_inv * A:")
print(identity2)
