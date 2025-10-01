# PSTH Analysis with digitalin.dat Support

This folder contains implementations for loading and analyzing digitalin.dat files directly, replacing the need for CSV-based interval data in PSTH analysis.

## Overview

Instead of using `pico_time_adjust.csv` files, this implementation loads interval data directly from `digitalin.dat` binary files using TTL event detection.

## Files

### Core Modules
- **`digitalin_loader.py`** - Main module for loading digitalin.dat files and extracting TTL events
- **`psth_digitalin_analysis.py`** - Modified PSTH analysis that supports both digitalin.dat and CSV loading
- **`test_digitalin_loading.ipynb`** - Jupyter notebook demonstrating the functionality

### Dependencies
Uses the existing utilities from the parent directory:
- `binary_data.py` - Binary data loading utilities
- `ttls.py` - TTL event detection and processing
- `intan.py` - Intan-specific data loading functions

## Key Features

### Channel Mapping
- **Channel 0**: Pico intervals (primary stimulus timing)
- **Channel 1**: Time markers (optional synchronization signals)

### Compatibility
- Drop-in replacement for existing CSV-based PSTH analysis
- Same DataFrame format as `pico_time_adjust.csv`
- Compatible with all existing PSTH analysis parameters

### Precision
- Full sampling rate precision (typically 30kHz)
- No conversion artifacts from CSV processing
- Direct binary data access

## Usage

### Basic Loading
```python
from digitalin_loader import load_digitalin_intervals

# Load intervals from digitalin.dat
intervals_df = load_digitalin_intervals(
    digitalin_filepath="path/to/digitalin.dat",
    sampling_rate=30000,
    pico_channel=0,    # Channel for pico intervals
    time_channel=1     # Channel for time markers
)
```

### PSTH Analysis
```python
from psth_digitalin_analysis import run_psth_analysis_digitalin

# Run PSTH analysis with digitalin.dat
results = run_psth_analysis_digitalin(
    unit=[1, 2, 3],
    duration=25,                    # ms
    digitalin_file="digitalin.dat",
    spikes_file="spikes.csv",
    use_digitalin=True,
    sampling_rate=30000
)
```

### Jupyter Notebook
Open `test_digitalin_loading.ipynb` for a complete walkthrough with examples and comparisons.

## Configuration

### File Paths
Update these paths in your analysis:
- **digitalin.dat**: Path to your binary digital input file
- **spikes.csv**: Path to your spike times file (same as before)

### Parameters
- **`sampling_rate`**: Recording sampling rate (default: 30000 Hz)
- **`pico_channel`**: Channel number for pico intervals (default: 0)
- **`time_channel`**: Channel number for time markers (default: 1)

### Channel Assignment
If your channel assignments differ:
```python
intervals_df = load_digitalin_intervals(
    digitalin_filepath="digitalin.dat",
    pico_channel=2,    # Change if pico is on different channel
    time_channel=3,    # Change if time is on different channel
    sampling_rate=30000
)
```

## Data Format

### Input: digitalin.dat
- Binary file with 16-bit digital input samples
- Each sample contains 16 channels (bits 0-15)
- Sampled at system rate (typically 30kHz)

### Output: DataFrame
Same format as `pico_time_adjust.csv`:
```
pico_Interval Start | pico_Interval End | pico_Interval Duration
(seconds)           | (seconds)         | (seconds)
```

## Validation

The loader includes validation to ensure compatibility:
- Checks required column names
- Validates data types and ranges
- Verifies timing consistency
- Reports duration statistics

## Comparison with CSV

### Advantages of digitalin.dat:
1. **Higher precision** - No CSV rounding errors
2. **Faster loading** - Direct binary access
3. **Original timing** - No conversion artifacts
4. **Flexible channels** - Configure any channel mapping

### CSV Fallback:
The analysis functions support both methods:
```python
# Use digitalin.dat
run_psth_analysis_digitalin(..., use_digitalin=True)

# Use CSV (fallback)
run_psth_analysis_digitalin(..., use_digitalin=False, intervals_file="pico_time_adjust.csv")
```

## Error Handling

The implementation includes robust error handling for:
- Missing files
- Invalid binary data
- TTL detection failures
- Channel configuration errors
- Sampling rate mismatches

## Testing

### Quick Test
```python
# Test with your data
python digitalin_loader.py
```

### Jupyter Notebook
Run `test_digitalin_loading.ipynb` for comprehensive testing and validation.

### Validation Steps
1. Load digitalin.dat file
2. Extract TTL events from both channels
3. Create compatible DataFrame
4. Compare with CSV results (if available)
5. Run PSTH analysis
6. Validate output plots

## Integration

### With Existing Pipeline
Replace CSV loading in your existing analysis:

```python
# Old way
intervals_df = pd.read_csv("pico_time_adjust.csv")

# New way
intervals_df = load_digitalin_intervals("digitalin.dat")
```

### Batch Processing
```python
# Process multiple units with digitalin.dat
results = run_psth_analysis_digitalin(
    unit=[1, 2, 3, 4, 5],
    duration=25,
    digitalin_file="digitalin.dat",
    spikes_file="spikes.csv",
    use_digitalin=True
)
```

## Troubleshooting

### Common Issues

**File not found**: Check absolute paths
```python
digitalin_file = os.path.abspath("path/to/digitalin.dat")
```

**No TTL events detected**: Check channel assignments
```python
# Try different channels
load_digitalin_intervals(..., pico_channel=1, time_channel=0)
```

**Wrong sampling rate**: Verify your recording parameters
```python
# Common rates: 20000, 30000, 40000 Hz
load_digitalin_intervals(..., sampling_rate=20000)
```

**Duration mismatch**: Check TTL transition direction
```python
# Auto-detect transition direction
load_digitalin_intervals(..., pico_transition=None)
```

### Debug Information
All functions provide detailed console output for debugging:
- File loading status
- Channel event counts
- Duration statistics
- Compatibility validation

## Future Enhancements

Potential improvements:
- Multi-channel interval detection
- Custom TTL threshold settings  
- Automatic channel detection
- Integration with other binary formats
- GUI for parameter configuration

## Contact

For questions or issues with the digitalin.dat loading functionality, refer to the existing spike analysis documentation or create an issue in the repository.