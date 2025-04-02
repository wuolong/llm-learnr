<https://deeplearning.cs.cmu.edu/S25/index.html>

# Lecture 01: Neural Network Introduction

## Breakthroughs in 2016

- voice recognition (Microsoft)
- translation (Google)
- AlphaGo defeated Lee Sedol in March 2016
- Caption generation of images

## Connectionist Machines

- Human makes connections between events
- Brain needs to store the associations and make inference
- *Mind and Body: The theories of their relation* by Alexander Bain (1873): information is stored in neural
  connections
- Neurons connect to other neurons. The processing/capacity of the brain is a function of these connections.
- Connectionist machines (neural networks) emulate this structure. In contrast, Von Neumann machines separate into
  Memory and Processor.

## Neural Networks

- Network of processing elements: knowledge is stored in the connection between the elements
- What are the elements? A neuron
- McCulloch and Pitt (1943) A Logical Calculus of the Ideas Immanent in Nervous Activity, Bulletin of Mathematical
  Biophysics, 5:115-137.
- Modeled the neurons of the brain (and the brain itself) as performing propositional logic, where each neuron
  evaluates the truth value of its input (propositions) - effectively Boolean logic

## Hebbian Learning

- If neuron $`x`$ repeatedly triggers neuron $`y`$ , the synaptic knob connecting to gets larger. Mathematically, the
  weight $`w`$ between the two neurons increases:
  ``` math
   w_{xy} = w_{xy} + \eta x y 
  ```
- Fundamentally unstable as there is no reduction in weights

## Rosenblatt Perceptron

- Frank Rosenblatt (1958) The Perceptron: a Probabilistic Model for Information Storage and Organization in the Brain,
  Psychological Review 65:398-408.
- Association units combine sensory input with fixed weights
– Response units combine associative units with learnable weights
  ``` math
    Y = 1 \text{if} \sum w_i x_i - T \geq 0 
  ```

- Learning rule: the weight is updated based on the error (difference between target and output) and input.
- A single perceptron is too week but a network of them (multi-layer perceptrons, or MLPs) can model arbitrarily
  complex Boolean functions.

## Perceptrons with Real Inputs

- Activation function $`\theta(z) = 1`$ if $`z >= 0`$, or 0 otherwise.  It can also output real value $`\sigma(z)`$.
- $`z`$ is an affine function of the inputs.
- A network of simple nodes can model complex decision rules.
- MLP can model a continuous function by combinations of (1/0) activators and weights.

# Lecture 02: Neural Nets As Universal Approximators

# Lecture 03: Training Neural Nets Part I

# Lecture 04: Training Neural Nets Part II

# Lecture 05: Training Neural Nets Part III

# Lecture 06: Training Neural Nets Part IV

# Lecture 07: Training Neural Nets Part V

# Lecture 08: Training Neural Nets Part VI
