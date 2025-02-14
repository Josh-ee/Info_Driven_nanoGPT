import os

def verify_even_distribution(filename, eval_spot=0.0):
    
    
    output_dir = os.path.dirname(__file__)
    file_path = os.path.join(output_dir, filename)
    
    
    with open(file_path, 'r') as f:
        equations = f.readlines()
    
    total = len(equations)
    if total == 0:
        print("No equations found in the file.")
        return
    
    eval_index = int(total * eval_spot)
    equations = equations[eval_index:]
    
    addition_count = sum(1 for eq in equations if '+' in eq)
    subtraction_count = sum(1 for eq in equations if '-' in eq)
    eval_total = len(equations)
    
    print(f"Evaluating from index {eval_index} onward ({eval_spot * 100:.1f}% of data)")
    print(f"Total equations in evaluation set: {eval_total}")
    print(f"Addition count: {addition_count}")
    print(f"Subtraction count: {subtraction_count}")
    
    if eval_total > 0:
        add_percentage = (addition_count / eval_total) * 100
        sub_percentage = (subtraction_count / eval_total) * 100
        print(f"Addition percentage: {add_percentage:.2f}%")
        print(f"Subtraction percentage: {sub_percentage:.2f}%")
        
        if abs(addition_count - subtraction_count) <= 1:
            print("The dataset has an even distribution of addition and subtraction equations.")
        else:
            print("The dataset does NOT have an even distribution of addition and subtraction equations.")
    else:
        print("No equations found in the evaluation range.")

# Specify the filename
filename = 'out_numbers.txt'

# Example evaluation spot
eval_spot = 0.71

# Run the verification function
verify_even_distribution(filename, eval_spot)
