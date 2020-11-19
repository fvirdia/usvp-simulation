"""
Script for generating TikZ plots representing the number of BKZ tours actually
run during Progressive BKZ.
"""


import json
from sage.all import save, line
from experimental_data_list import pbkz_data


plot_head = """    \\begin{tikzpicture}
    \\begin{axis}[
        /pgf/number format/.cd,fixed,,
        xlabel=round using block size $\\beta$,
        width=\\columnwidth,
        height=0.4\\columnwidth,
        legend cell align=left,
        legend pos=south east,
        ytick={%s},
        yticklabels={%s}
    ]

    \\addplot[blue] coordinates {
        """
plot_foot = """
    };
    \\addlegendentry{%s};
    \\end{axis}
    \\end{tikzpicture}
"""

data = pbkz_data("full-lll-tour_map-skip-1")
# data = pbkz_data("full-lll-bu-tour_map-skip-1")

for record in data:
    tour_map_fn = record[1]
    n, q, sd, m, _, tau, skip, _, _ = record[-1]
    print(record[-1])
    tour_map = json.load(open(tour_map_fn))
    tour_map = tour_map[list(tour_map.keys())[0]]
    tour_out = {}
    for k in tour_map:
        tour_out[int(k)] = tour_map[k][0]

    if tau == 5:
        ticks = range(1, 6)
    elif tau == 10:
        ticks = range(2, 11, 2)
    elif tau == 15:
        ticks = [1] + list(range(3, 16, 3))
    elif tau == 20:
        ticks = [1] + list(range(4, 21, 4))

    ticks = ",".join(list(map(str, ticks)))
    plt = plot_head % (ticks, ticks)
    plt += " ".join(list(map(str, sorted(tour_out.items()))))
    plt += plot_foot % "tours completed before auto-abort"
    with open(f"plots/tour_maps/n{n}-tau{tau}-skip{skip}.tex", 'w') as f:
        f.write(plt)
