# Nanorings

## Preliminary
The default device is set by importing `DEVICE` from `neural_dynamics.config`. The device is selected from the first one of the following: `mps`, `cuda`, and `cpu`.

> [!WARNING]
> The code is only tested on `mps`. Issues may be present when training on `cuda` or `cpu`.

## Training workflow

### Load the training data
The training data is loaded

```python
trajectories, time_grid, h_signal, metadata = load_dataset(config.dataset_path)
```

> [!IMPORTANT]
> `metadata` may be deprecated in the future, it exists only for debugging

The `load_dataset` class is defined as

```python
def load_dataset(dataset_path: Path) -> Tuple[Tensor, Tensor, Tensor, Dict[str, Any]]:
    """Load the nanorings dataset from a .pt file.
    
    Args:
        dataset_path: Path to the cached dataset file.
        
    Returns:
        Tuple of (trajectories, time_grid, h_signal, metadata) where:
        - trajectories: [num_runs, num_steps, state_dim] AMR responses
        - time_grid: [num_steps] time axis
        - h_signal: [num_steps] exogenous H-field forcing
        - metadata: Dataset information (empty dict if not present in .pt file)
    """
    LOGGER.info("Loading dataset from %s", dataset_path)
    cached: MutableMapping[str, Any] = torch.load(dataset_path)

    trajectories = cached["trajectories"].float()
    time_grid = cached["time_grid"].float()
    h_signal = cached["h_signal"].float()
    metadata = dict(cached.get("metadata", {}))

    LOGGER.info("Trajectories shape: %s", tuple(trajectories.shape))
    LOGGER.info("Time grid shape: %s", tuple(time_grid.shape))
    LOGGER.info("H-field shape: %s", tuple(h_signal.shape))
    LOGGER.info("Metadata: %s", metadata)

    return trajectories, time_grid, h_signal, metadata
```

### Pre-process training data
For the nanorings example, we pre-process the trajectories in the main script for easier experimentation. 

> [!IMPORTANT]‌
> In a future release, pre-processing will be performed in data/nanorings/preprocess.py` and not in this script.

The pre-processing converts long trajectories into chunked windows, resamples the trajectories to `CONFIG.TARGET_TIMESTEPS` using the function `resample_to_target_tiomesteps` by a linear interpolation to ensure that the trajectories contain a consistent number of time steps, and standardises the data such that there are no offsets in the data and that the mean and standard deviation are 0 and 1.

```python
trajectories, time_grid, h_signal, standardisation_stats = preprocess_nanorings_trajectories(
    trajectories,
    time_grid,
    h_signal,
    config,
    metadata,
)
```

### Fetch hyperparameters and network configuration
We instantiate the `NanoringsHyperparameters()` class from `examples/systems/parameters/nanorings.py`

```python
config = NanoringsHyperparameters()
```

This class is frozen (immutable).

### Build the model architecture 
Using `config`, 