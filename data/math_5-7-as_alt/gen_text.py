import random
import re

# Helper function to extract the unordered pair (as a tuple) from an equation string.
def extract_pair(eq):
    # Equation is like "<3+9=12>" or "<3-9=-6>"
    content = eq.strip('<>')
    # Use regex to capture the two numbers (before and after the operator)
    m = re.match(r'(\d+)[+-](\d+)=', content)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return tuple(sorted((a, b)))
    return (0, 0)  # fallback if not matching

# This function takes a list of equations and “distributes” them evenly into num_quarters buckets.
# It sorts by the pair (so that equations with the same numbers are together), then assigns
# them round-robin so that (for example) if there are two equations with the pair (3,9)
# they end up in different quarters.
def distribute_evenly(equations, num_quarters=4):
    # Sort by pair key to group similar equations together.
    equations.sort(key=lambda eq: extract_pair(eq))
    # Create empty buckets for each quarter.
    quarters = [[] for _ in range(num_quarters)]
    # Assign each equation to a bucket in round-robin order.
    for idx, eq in enumerate(equations):
        quarters[idx % num_quarters].append(eq)
    # Shuffle each quarter (to avoid any ordering artifacts)
    for q in quarters:
        random.shuffle(q)
    # Concatenate the quarters back together.
    distributed = []
    for q in quarters:
        distributed.extend(q)
    return distributed

# Function to generate addition and subtraction equations.
def generate_equations(skip=None):
    addition_equations = []
    subtraction_equations = []
    skip = skip or []  # default to empty list if no skip values passed
    
    for i in range(1, 10):  # Only positive numbers
        for j in range(1, 10):
            if (i, j) not in skip and (j, i) not in skip:
                # Generate addition equation
                add_eq = f"<{i}+{j}={i + j}>"
                if len(add_eq) == 7:
                    add_eq += " "  # pad if necessary
                addition_equations.append(add_eq)
                
                # Generate subtraction equation (allowing negative results)
                sub_eq = f"<{i}-{j}={i - j}>"
                if len(sub_eq) == 7:
                    sub_eq += " "
                subtraction_equations.append(sub_eq)
    
    return addition_equations, subtraction_equations

# Updated merge: first “evenly distribute” each operator's equations,
# then interleave them so that the final list alternates addition and subtraction.
def merge_equations(additions, subtractions):
    even_additions = distribute_evenly(additions, num_quarters=4)
    even_subtractions = distribute_evenly(subtractions, num_quarters=4)
    
    final_equations = []
    # Interleave the two lists.
    for a, s in zip(even_additions, even_subtractions):
        final_equations.append(a)
        final_equations.append(s)
        
    # If one list is longer, append the remaining equations.
    if len(even_additions) > len(even_subtractions):
        final_equations.extend(even_additions[len(even_subtractions):])
    elif len(even_subtractions) > len(even_additions):
        final_equations.extend(even_subtractions[len(even_additions):])
    
    return final_equations

# Function to write the equations to a text file.
def write_equations_to_file(filename, skip=None):
    additions, subtractions = generate_equations(skip)
    # Merge the two lists with alternating operators and even distribution.
    equations = merge_equations(additions, subtractions)
    
    with open(filename, 'w') as f:
        for equation in equations:
            f.write(equation + '\n')

# Specify the filename and numbers to skip (e.g., skipping (5,7) and (7,5))
filename = 'out_numbers.txt'
skip_numbers = [(5, 7), (7, 5)]

# Write the equations to file.
write_equations_to_file(filename, skip=skip_numbers)

print(f"Equations written to {filename} with alternating '+' and '-' and an even distribution of number conversations.")
