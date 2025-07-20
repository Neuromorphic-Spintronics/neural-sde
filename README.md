# Neural SDE Framework

> [!NOTE]
> This README is---for the most part---a plan for the code implementation. Some functions are partially or not yet implemented, and are labelled accordingly.

A PyTorch framework for training neural stochastic differential equations (SDEs) to model dynamical systems with both deterministic and stochastic components. This framework enables the learning of dynamical systems through a combination of neural networks representing drift and diffusion terms, with adversarial training.

## Overview

The neural SDE framework provides a complete pipeline for:

- **System Modelling**: Defining dynamical systems with both deterministic and stochastic components
- **Neural Network Architecture**: Learning drift and diffusion terms using neural networks
- **Numerical Integration**: Robust integration methods for both ODEs and SDEs
- **Training Pipeline**: Multi-phase training with optional adversarial components
- **Analysis & Visualisation**: Comprehensive tools for system analysis and result visualisation

## Architecture Overview

The codebase is organised into several key components that work together to provide a complete neural SDE framework:

### 1. Configuration & Device Management (`config.py`)

The framework automatically selects the optimal computation device:

- **Apple Silicon**: Uses MPS (Metal Performance Shaders) for GPU acceleration
- **NVIDIA GPUs**: Uses CUDA for GPU acceleration  
- **Fallback**: Uses CPU for universal compatibility

```python
from config import DEVICE
# All tensors and models automatically use the optimal device
```

### 2. Parameter Management (`parameters/`)

The parameter system provides structured configuration for all components:

#### `hyperparameters.py`

- **`NetworkArchitecture`**: Defines neural network layer structures with validation
- **`LearningRates`**: Configures learning rates for different network components
- **`Hyperparameters`**: Complete training configuration including system dimensions

#### System-Specific Parameters

- **`duffing_oscillator.py`**: Parameters for the stochastic Duffing oscillator
- **`leaky_integrator.py`**: Parameters for leaky integrator systems

> [!NOTE]
> Only `duffing_oscillator.py` is implemented.

### 3. Dynamical Systems (`systems/`)

The systems module provides reference implementations of dynamical systems for training data generation:

#### `duffing_oscillator.py`
Implements the stochastic Duffing oscillator:
```
d^x/dt^2 + delta * dx/dt + alpha * x + beta * x^3 = F(t) + eta(t) + xi(t).
```

**Key Features:**
- **Deterministic Component**: Linear (alpha) & non-linear (beta) restoring forces with damping (delta)
- **Stochastic Components**: White noise (eta) and coloured noise (xi) via Ornstein-Uhlenbeck process
- **External Forcing**: (e.g., a periodic) forcing function F(t)
- **Energy Analysis**: Potential, kinetic, and total energy computation

**Usage:**
```python
from systems.duffing_oscillator import DuffingOscillator
from parameters.duffing_oscillator import DuffingOscillatorParameters

params = DuffingOscillatorParameters(...)
system = DuffingOscillator(params)
time_grid, trajectory = system.integrate_sde()
```

### 4. Numerical Integration (`models/integrators.py`)

The integration module provides both full trajectory-based and single-step integrators, using PyTorch tensors throughout:

#### Trajectory Integrators (for training data generation)

- **`rk2()`**: Second-order Runge-Kutta for ODEs
- **`rk4()`**: Fourth-order Runge-Kutta for ODEs
- **`euler_maruyama()`**: First-order SDE integration
- **`stochastic_heun_method()`**: Second-order SDE integration (predictor-corrector)

#### Single-Step Integrators (for neural SDE training)

- **`euler_maruyama_step()`**: Single step for both ODEs and SDEs
- **`stochastic_heun_step()`**: Improved accuracy for SDEs
- **`runge_kutta_4_step()`**: High-order accuracy for ODEs

> [!CAUTION]
> The full trajectory integrators were tested in a previous version, but their new impelemntation via the single-step integrators need unit tests written.

### 5. Neural SDE Framework (`models/neural_sde.py`)

The core neural SDE implementation combines neural networks with numerical integration:

#### Network Components

- **`DriftNet`**: Learns deterministic dynamics mu(x, t, u)
- **`DiffusionNet`**: Learns stochastic dynamics sigma(x, t, u)
- **`DiscriminatorNet`**: Adversarial critic for Wasserstein GAN training

> [!CAUTION]
> Not Yet Implemented

**Usage:**
```python
from models.neural_sde import NeuralSDE
from parameters.hyperparameters import Hyperparameters

hyperparams = Hyperparameters(...)
neural_sde = NeuralSDE(hyperparams)

# Simulate trajectory
inputs = torch.randn(batch_size, input_dim, num_timesteps)
initial_state = torch.randn(batch_size, state_dim)
trajectory = neural_sde(inputs, initial_state=initial_state)
```

### 6. Training Pipeline (`training/`)

The training module provides components for model training:

- **`train_loop.py`**: Main training loop implementation
- **`loss_functions.py`**: Loss function definitions for different training phases
- **`evaluation.py`**: Model evaluation and validation utilities

### 7. Utilities (`utils/`)

#### `plotting.py`

Provides visualisation tools:

- **`set_default_plotting_style()`**: Configures matplotlib for a consistent plotting aesthetic
- **`style_axis_clean()`**: Clean axis styling
- **Colour Palette**: Consistent colour scheme across all visualisations

#### `logging.py`

Structured logging for training and evaluation processes. 

> [!CAUTION]
> Not Yet Implemented

## Data Flow Through the Framework

### 1. System Definition & Data Generation

```python
# Define system parameters
params = DuffingOscillatorParameters(...)

# Create system instance
system = DuffingOscillator(params)

# Generate training data
time_grid, trajectory = system.integrate_sde() # stochastic Heun is chosen by default
```

### 2. Neural Network Configuration

```python
# Define network architectures
drift_architecture = NetworkArchitecture(input_size=4, hidden_sizes=[64, 64], output_size=2)
diffusion_architecture = NetworkArchitecture(input_size=4, hidden_sizes=[32, 32], output_size=4)

# Create hyperparameters
hyperparams = Hyperparameters(
    drift_network=drift_arch,
    diffusion_network=diffusion_architecture,
    state_dimension=2,
    input_dimension=1,
    timestep=0.01
)
```

### 3. Model Initialisation & Training

```python
# Create neural SDE model
neural_sde = NeuralSDE(hyperparams)

# Phase 1: Train deterministic dynamics
# (diffusion_net=None during this phase)

# Phase 2: Enable stochastic mode
neural_sde.enable_stochastic_mode(noise_dimension=1)

# Phase 3: Enable adversarial training (if using SDE)
neural_sde.enable_adversarial_training(trajectory_length=50)
```

> [!CAUTION]
> Not Yet Implemented

### 4. Simulation & Analysis

```python
# Generate external inputs
inputs = torch.randn(batch_size, input_dim, num_timesteps)

# Simulate with neural SDE
trajectory = neural_sde(inputs, initial_state=initial_state)

# Analyse results
position, velocity = system.get_position_velocity(trajectory)
energy = system.total_energy(position, velocity)
```

> [!CAUTION]
> Not Yet Implemented

## Development Tools

The framework uses modern Python development tools:

- **`uv`**: Fast package management
- **`ruff`**: Fast linting
- **`ty`**: Type checking (pre-release)
- **`pytest`**: Testing with coverage
- **`zarr`**: Efficient data storage for large datasets

## Getting Started

1. **Install Dependencies**:

   ```bash
   uv sync
   ```

2. **Run Tests**:

   ```bash
   pytest tests/
   ```

3. **Generate Example Data**:

   ```python
   from examples.duffing_oscillator import run_duffing_example
   run_duffing_example()
   ```

  > [!CAUTION]
  > Not Yet Implemented