

# Phase and Magnitude Alignment (PMA)

PMA: a novel method for time series alignment and distance measure. 

* PMA aligns time series through segment-to-segment matching, making it robust to local phase and magnitude variations. 

* PMA separately measures phase and magnitude differences, while providing shape distance invariant to local scaling and shifting through segment-wise alignment. 

## Requirements

1. Install Python 3.7, and the required dependencies.
2. Required dependencies can be installed by: ```pip install -r requirements.txt```

## Data

The UCR datasets can be obtained at https://www.cs.ucr.edu/~eamonn/time_series_data_2018/.

## Usage

```python
from PMA.dPMA import pma
aligned_segments, distances, original_information, alignment_pairs = pma(samp, proto)
```

Parameters:

```python
samp: tensor(batch_size, num_var, len_seq)
proto: tensor(batch_size, num_var, len_seq)
```

returns:

```
aligned_segments: list of K aligned segments
distances, list of K phase, magnitude, and shape distance
original_information, list of K original phase and magnitude information
alignment_pairs, [(0,0), ..., (len_samp, len_proto)]
```



