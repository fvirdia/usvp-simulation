# uSVP Simulation -- Simulating the success probability of solving the unique Shortest Vector Problem

This repository provides the code and data release for 
>  _On the Success Probability of Solving Unique SVP via BKZ_  
>  Eamonn W. Postlethwaite, Fernando Virdia, IACR ePrint [2020/1308](https://eprint.iacr.org/2020/1308)

This release includes
- the code used to simulate the success probability of BKZ and Progressive BKZ for solving uSVP,
- the code used to run the experiments,
- the raw results from the experiments,
- the code for generating plots and tables in the paper.

## Requirements

- Sagemath 9.0+
- git (optional, for getting a copy of the [DSDGR20] simulator and the [APS15] LWE estimator)
- make (optional, simplifies setup)
- pdflatex (optional, for generating plots using Tikz)

The code was written and tested on Linux. It should also work fine on WSL for Windows and macOS.

### Buggy FPLLL release in Sagemath 9.0 binaries

The binary distribution of Sagemath 9.0 contains a bug in their FPLLL internal library. To successfully run our experiments, this should be fixed by reinstalling FPLLL and FPyLLL as described in paragraph "Manual update of fpylll and fplll inside Sagemath 9.0+" of the FPylll [readme](https://github.com/fplll/fpylll/blob/master/README.rst).

## Instructions for reproducing plots and tables from raw data

To create the directory structure where the plots and tables will be located, run

```
make setup
```

To reproduce all plots/tables in the paper at once (this takes a while), run

```
make all
``` 

Plots will be located under `/plots/`. The state of this directory can be reset running

```
make clean
```

Results for tables will be found printed to terminal. By default we omit generating Figure 1, since it takes significantly the longest. It can be re-enabled by uncommenting the call to `compare_vs_lwe_side_channel` in `reproduce.py`.

To reproduce individual plots/tables, run
- `sage testing-z-shape.py`, to generate Figure 11 (in `/plots/qary-lll-sim-n100.pdf`)
- `sage tours_actually_run.py`, to generate Figure 7 (by combining `n100-tau5-skip1.tex`, `n100-tau10-skip1.tex`, `n100-tau15-skip1.tex`, `n100-tau20-skip1.tex` in `/plots/tour_maps/`)
- `sage explain_why_more_security.py`, to generate examples of the difference caused by using the GSA or the [CN11] simulator in the LWE Estimator [APS15] (Figure 10 in `/plots/lwe-estimator-with-cn11/`)
- `sage stddev_in_practice.py`, to generate the variance of expected sample variance for the three distributions considered in our paper
- `sage estimates.py`, to generate the LWE Estimator's numbers in Table 2
- `sage reproduce.py`, to generate all other Figures and the simulation numbers in Table 2
- `sage reproduce.py --tikz` will do the same as above, but use PdfLatex to generate the plots.

### LWE Estimator using the [CN11] simulator

While these numbers are not reported in Figures or Tables, it is possible to tweak
the LWE estimator to use the [CN11] simulator as described in footnote 6.

To enable that modification, run `make aps15`, then set `use_sim_for_beta = True` on line 2103 of `lwe-estimator/estimator.py`.

## Instructions for reproducing experiments

The main module for reproducing the experiments is `experiments.py`.
It requires some tweaking to reproduce some of the experiments. 
To run an example set of parameters (listed in the `parameter_sets` variab), is is sufficient to rung `sage run_experiments.py`. 
This will run Progressive BKZ experiments by default. These can be turned into BKZ experiments by commenting/uncommenting the appropriate lines in `run_experiments.py`.

## References

[APS15] Martin R Albrecht, Rachel Player, and Sam Scott. _On the concrete hardness of learning with errors_. JMC, 2015.

[CN11] Yuanmi Chen and Phong Q Nguyen. _Bkz 2.0: Better lattice security estimates_. In ASIACRYPT, 2011.

[DSDGR20] Dana Dachman-Soled, Léo Ducas, Huijing Gong, and Mélissa Rossi. _LWE with side information: Attacks and concrete security estimation_. In
CRYPTO, 2020.
