import os

def create_files(n):
    if not (1 <= n < 100):
        print("Error: n must be between 1 and 99.")
        return
    
    template = """import numpy as np
import matplotlib.pyplot as plt


def main():
    pass


if __name__ == '__main__':
    main()
"""
    
    for i in range(1, n + 1):
        filename = f"task{i:02d}.py"  # Generates task01.py, task02.py, ..., taskn.py
        with open(filename, "w") as file:
            file.write(template)
        print(f"Created {filename}")

if __name__ == "__main__":
    try:
        n = int(input("Enter the number of files to create (n < 100): "))
        create_files(n)
    except ValueError:
        print("Invalid input. Please enter an integer.")
