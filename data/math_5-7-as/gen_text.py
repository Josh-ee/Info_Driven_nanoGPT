import random

# Function to generate addition and subtraction equations
def generate_equations(skip=None):
    equations = []
    skip = skip or []  # Default to an empty list if no skip values are passed
    
    for i in range(1, 10):
        for j in range(1, 10):
            if (i, j) not in skip and (j, i) not in skip:  # Skip the pair (i, j) and (j, i)
                # Addition equation
                addition_equation = f"<{i}+{j}={i + j}>"
                if len(addition_equation) == 7:
                    addition_equation += " "
                equations.append(addition_equation)
                
                # Subtraction equation (ensuring non-negative result)
                if i >= j:
                    subtraction_equation = f"<{i}-{j}={i - j}>"
                else:
                    subtraction_equation = f"<{j}-{i}={j - i}>"
                if len(subtraction_equation) == 7:
                    subtraction_equation += " "
                equations.append(subtraction_equation)

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
filename = 'addition_and_subtraction.txt'

# Specify the numbers to skip (for example, skip 5 and 7)
skip_numbers = [(5, 7), (7, 5)]  # This will skip both <5+7=12> and <7+5=12>

# Call the function with randomization enabled and skipping the specified numbers
write_equations_to_file(filename, randomize=True, skip=skip_numbers)

print(f"Equations written to {filename} with specified numbers skipped.")
