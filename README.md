# Machine Learning for Integration and Differentiation
A trained model that solve beginner integration and differentiation problems
##  Calculus Problem Generator & Solver ML Project

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![SymPy](https://img.shields.io/badge/SymPy-Mathematics-green)


An end-to-end system for generating calculus problems and training ML models to solve them.

##  Key Features
- **Problem Generation**: Creates 2,000+ calculus problems with verified solutions
- **Dual Mode**: Handles both differentiation and integration
- **Difficulty Levels**: From basic to advanced problems
- **T5 Model**: Fine-tuned transformer for symbolic math


##  Quick Start

### 1. Generate Problems

from sympy import symbols
from problem_generator import generate_advanced_problem

x = symbols('x')
problem = generate_advanced_problem()  # Returns dict with problem/solution


### 2. Train Model
from transformers import T5ForConditionalGeneration

model = T5ForConditionalGeneration.from_pretrained("t5-small")
# See notebook for full training code

### 3. Predict
# After training:
solver("diff: x^2 + 3x")  # Returns "2x + 3"
solver("int: cos(x)")     # Returns "sin(x) + C"

## Development Setup 

### Install dependencies
pip install sympy matplotlib transformers torch

run mlipynb.py
