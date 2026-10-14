# Data

This study uses two scenarios of the public **DeepSense 6G** dataset. The recordings are
third-party and are **not** redistributed here.

## What to obtain

Scenarios **31** and **33** of DeepSense 6G Testbed 5, from
<https://www.deepsense6g.net/>. Both are roadside deployments in which

* unit 1 is **stationary** and carries a 60 GHz SIVERS receiver with a 16-element uniform
  linear array and a 64-beam combining codebook over a 90 degree field of view, **and** a
  separate TI FMCW radar at 76 to 81 GHz;
* unit 2 is a vehicle carrying a 60 GHz quasi-omnidirectional transmitter;
* every modality is logged at 10 Hz.

The radar cube is `(4 receive channels, 256 fast-time samples, 250 chirps)` per frame.

## Where to put it

Place the extracted scenarios so that the following resolve:

```
datasets/deepsense6g/Scenario31/scenario31_dev.csv
datasets/deepsense6g/Scenario33/scenario33_dev.csv
```

with the per-frame radar cubes and `unit1_pwr_60ghz` power files at the relative paths the
CSVs name. Nothing else is required: every script reads the CSVs and follows their paths.

## What the scripts need

Only the columns `unit1_beam`, `unit1_pwr_60ghz`, `unit1_radar`, `unit1_loc`, `unit2_loc`,
`seq_index` and the timestamps. The 60 GHz power file gives the received power of all 64
beams per frame; `unit1_beam` is its argmax, which the loaders verify on every frame.

## Size

The two scenarios together are a few hundred gigabytes extracted. The preprocessing caches
the scripts build (`experiments/cache/`) are a further few gigabytes and are regenerated
rather than versioned.
