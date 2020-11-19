"""
Example file for using the simulators, and comparing their output.
Run `sage test.py`, and a png will be output.
"""


from math import log
from sage.all import line, save
from fpylll import IntegerMatrix, GSO, LLL, FPLLL, BKZ
from fpylll.tools.bkz_simulator import simulate as CN11_simulate
import CN11 # this is identical to fpylll.tools.bkz_simulator
import BSW18

n_halfs, block_size, max_loops = 75, 60, 1000

# generate lattice instance
FPLLL.set_random_seed(1337)
q = 2**10
mat = IntegerMatrix.random(2*n_halfs, "qary", q=q, k=n_halfs)
A = LLL.reduction(mat)
M = GSO.Mat(A)
M.update_gso()

# print("Input gso norms, natural log")
# print(list(map(lambda x: log(x)/2, M.r())))

fplll_cn11 = CN11_simulate(M, BKZ.Param(block_size=block_size, max_loops=max_loops))
cn11 = CN11.simulate(M, BKZ.Param(block_size=block_size, max_loops=max_loops))
bsw18 = BSW18.simulate(M, BKZ.Param(block_size=block_size, max_loops=max_loops))
absw18 = BSW18.averaged_simulate(M, BKZ.Param(block_size=block_size, max_loops=max_loops))

g = line([])
g += line(
        [(i, log(fplll_cn11[0][i])/2) for i in range(len(cn11[0]))],
        legend_label="CN11",
        title="β = %d, n = %d, loops = %d" % (block_size, 2*n_halfs, max_loops)
    )
g += line(
        [(i, log(cn11[0][i])/2) for i in range(len(cn11[0]))],
        color="black",
        linestyle="--",
        legend_label="CN11",
    )
g += line(
        [(i, log(bsw18[0][i])/2) for i in range(len(bsw18[0]))],
        color='red',
        legend_label="BSW18"
    )
g += line(
        [(i, log(absw18[0][i])/2) for i in range(len(absw18[0]))],
        color='orange',
        legend_label="averaged BSW18"
    )
save(g, "beta_%d.png" % block_size, dpi=300)
