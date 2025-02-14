import random

# Function to generate addition and subtraction equations separately
def generate_equations(skip=None):
    addition_equations = []
    subtraction_equations = []
    skip = skip or []  # Default to an empty list if no skip values are passed
    
    for i in range(1, 10):  # Only positive numbers
        for j in range(1, 10):  # Only positive numbers
            if (i, j) not in skip and (j, i) not in skip:
                # Generate addition equation
                add_eq = f"<{i}+{j}={i + j}>"
                if len(add_eq) == 7:
                    add_eq += " "
                addition_equations.append(add_eq)
                
                # Generate subtraction equation (allowing negative results)
                sub_eq = f"<{i}-{j}={i - j}>"
                if len(sub_eq) == 7:
                    sub_eq += " "
                subtraction_equations.append(sub_eq)
    
    return addition_equations, subtraction_equations

# Function to merge the two lists randomly into a single list with even distribution
def merge_equations(additions, subtractions, randomize=True):
    total_count = len(additions) + len(subtractions)
    result = [None] * total_count
    
    # Create a list of positions [0, 1, ..., total_count-1]
    positions = list(range(total_count))
    
    # Randomly choose positions for the addition equations
    add_positions = random.sample(positions, len(additions))
    # The remaining positions will be used for the subtraction equations
    sub_positions = [pos for pos in positions if pos not in add_positions]
    
    if randomize:
        random.shuffle(additions)
        random.shuffle(subtractions)
    
    # Place addition equations into their randomly chosen positions
    for pos, eq in zip(add_positions, additions):
        result[pos] = eq
        
    # Place subtraction equations into the remaining positions
    for pos, eq in zip(sub_positions, subtractions):
        result[pos] = eq
        
    return result

# Function to write equations to a text file
def write_equations_to_file(filename, randomize=True, skip=None):
    additions, subtractions = generate_equations(skip)
    # Merge the two lists to ensure an even, random distribution
    equations = merge_equations(additions, subtractions, randomize)
    
    with open(filename, 'w') as f:
        for equation in equations:
            f.write(equation + '\n')

# Specify the filename and numbers to skip (e.g., skipping (5,7) and (7,5))
filename = 'out_numbers.txt'
skip_numbers = [(5, 7), (7, 5)]

# Write the equations to file with random even distribution
write_equations_to_file(filename, randomize=True, skip=skip_numbers)

print(f"Equations written to {filename} with an even random distribution of '+' and '-'.")
