import random

# Function to generate the equations
def generate_equations(skip=None):
    equations = []
    skip = skip or []  # Default to an empty list if no skip values are passed
    for i in range(1, 10):
        for j in range(1, 10):
            if (i, j) not in skip and (j, i) not in skip:  # Skip the pair (i, j) and (j, i)
                equation = f"<{i}+{j}={i + j}>"
                if len(equation) == 7:
                    equation += " "
                equations.append(equation)
    return equations

# Function to write equations to a text file
def write_equations_to_file(filename, randomize=False, skip=None):
    equations = generate_equations(skip)
    
    if randomize:
        random.shuffle(equations)
    
    with open(filename, 'w') as f:
        for equation in equations:
            f.write(equation + '\n')

# Specify the filename
filename = 'equations.txt'

# Specify the numbers to skip (for example, skip 5 and 7)
skip_numbers = [(5, 7), (7, 5)]  # This will skip both <5+7=12> and <7+5=12>

# Call the function with randomization enabled and skipping the specified numbers
write_equations_to_file(filename, randomize=True, skip=skip_numbers)

print(f"Equations written to {filename} with specified numbers skipped.")
