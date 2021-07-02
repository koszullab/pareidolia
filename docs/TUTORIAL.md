# Tutorial

## Detection

The simplest way to run pareidolia is to specify input files, their corresponding conditions (in the same order) and the output file:

```bash
pareidolia ctrl1.cool,ctrl2.cool,treat1.cool,treat2.cool control,control,treated,treated treatment_loops.tsv
```

Which will detect differential loops between the two conditions and return their coordinates and differential score (treated - control). The condition of the first input file is assumed to be the control and scores will be relative to this condition.

## Quantification

In some cases, the set of loops is already known and we just need a differential score for those coordinates. This can be done by specifying the `--bed2d-file` option. When this option is provided, pareidolia skips detection and returns scores for each input position.

```bash
# Compute differential loop scores at predefined positions.
# Loops appearing upon treatments have positive values and vice versa
pareidolia --bed2d-file my_loops.bed2d \
           ctrl1.cool,ctrl2.cool,treat1.cool,treat2.cool \
           control,control,treated,treated \
           known_loops_change.tsv
```

## Additional options

Pareidolia applies 3 filters to reduce spurious detections and only return high confidence scores. Each of those filters is controlled by a command line option:

* `--pearson`: First, only coordinates where at least one sample has a high Chromosight scores are considered
* `--snr`: Then, regions with a signal-to-noise-ratio above a certain value are kept. This is defined as the pixel-wise Chromosight score difference between conditions divided by the standard deviation within conditions.
* `--density`: Finally, pixels of low contact density are filtered out. Contact density is defined as the proportion of nonzero pixel in the window used to compute the Chromosight score.

In some cases, it can be useful to skip those filters altogether when running in quantification mode (with `--bed2d`) to retrieve scores for all input coordinates. For those cases, the `--no-filter` option will skip all 3 filters.

The analysis can also be sped up by restricting computations to a chromosome or region of interest (e.g. 'chr1' or 'chr1:13000-20000'). It is also possible to exploit multiple cpu cores using `--n-cpus` (e.g. `--n-cpus 4` to run with 4 cpus in parallel). This will analyze samples in parallel on different CPUS, which means that memory requirements will increase linearly with the number of CPUs specified and that there can be no more parallel tasks than there are samples to process.

## Custom kernels

Standard patterns provided by chromosight can be provided as strings given to the `--kernel`. The default value is `loops`, but we can detect domain borders instead by specifying `--kernel borders`. For those standard Chromosight kernels, default parameter values for `--max_dist` and `--pearson` will be adapted based on the pattern.

It is also possible to provide a custom kernel matrix in the form of a tab-separated text file containing a square numeric matrix with an odd number of rows. This also means that you should specify `--pearson` and `--max-dist` values for the custom kernel.

```bash
# Generate dummy kernel matrix
echo "0\t1\t0\n0\t1\t0\n0\t1\t0" > mat.tsv
pareidolia --kernel mat.tsv \
           --pearson 0.3 \
           --max-dist 300000 \
           ctrl1.cool,ctrl2.cool,treat1.cool,treat2.cool \
           control,control,treated,treated \
           treatment_custom.tsv
```