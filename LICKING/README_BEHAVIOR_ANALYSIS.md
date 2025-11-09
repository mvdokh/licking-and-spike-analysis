
 `Phox2B#8ALL` 

 `behavior.csv` from `720_1`, `725_2`, etc.

pandas numpy matplotlib seaborn

`/home/wanglab/spike-analysis/LICKING/OUTPUT/`


1. **licks_per_bout.png**
   - Distribution of licks per bout across sessions (box plot)
   - Mean licks per bout with error bars

2. **max_area_per_lick.png**
   - Distribution of maximum tongue area per lick (violin plot)
   - Mean max tongue area across sessions

3. **temporal_trends.png**
   - Total licks (normalized) per day
   - Mean licks per bout over time
   - Mean max tongue area over time
   - Mean lick duration over time

4. **bout_characteristics.png**
   - Distribution of bout durations
   - Relationship between licks per bout and mean tongue area
   - Normalized bouts per session (bouts divided by video frame count)
   - Distribution of lick rates within bouts

5. **heatmap_analysis.png**
   - Heatmap of mean max tongue area across sessions
   - Heatmap of mean licks per bout across sessions

6. **bout_length_comparison.png**
   - Distribution comparison of bout lengths between early and late sessions (histogram overlay)
   - Box plot comparison of bout lengths (early vs late)
   - Cumulative distribution function (CDF) of bout lengths
   - Trend in proportion of long bouts over time

7. **max_bout_length_over_time.png**
   - Maximum bout length for each session (color-coded bar chart)
   - Bout length range showing min-max spread with percentiles
   - Maximum bout length over time with smoothed trend line


- **session_summary.csv**: Detailed statistics for each session
- **overall_summary.txt**: Overall summary statistics


```
Phox2B#8ALL/
├── 720_1/
│   └── behavior.csv
├── 725_1/
│   └── behavior.csv
├── 725_2/
│   └── behavior.csv
└── ...
```

- Format: `{date}_{trial_number}`
- Example: `720_1` = July 20, Trial 1


- `Tongue_area_Interval Max`: Maximum area of the tongue for a lick
- `Tongue_area_interval_detection_Interval Start`: Lick start frame
- `Tongue_area_interval_detection_Interval End`: Lick end frame
- `Tongue_area_interval_detection_Interval Duration`: Duration of a lick
- `Tongue_area_interval_detection_group_intervals_Interval Overlap Assign ID`: Bout ID




To modify the base directory, edit line 366 in `behavior_analysis.py`:

```python
base_dir = "/mnt/c/Users/wanglab/Desktop/Phox2B#8ALL"
```

To change the output directory, modify the `__init__` method of the `BehaviorAnalyzer` class (line 21):

```python
self.output_dir = Path("/home/wanglab/spike-analysis/LICKING/OUTPUT")
```

