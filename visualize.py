
from typing import Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
import random

def draw_network(genome, fname=None):
    # simple layered layout by node id (not real topo sort but stable)
    xs, ys = {}, {}
    ins = [i for i, n in genome.nodes.items() if n.type=='in']
    outs = [i for i, n in genome.nodes.items() if n.type=='out']
    hids = [i for i, n in genome.nodes.items() if n.type=='hid']

    def place(nodes, y):
        for k, nid in enumerate(sorted(nodes)):
            xs[nid] = k
            ys[nid] = y

    place(ins, 0)
    place(hids, 1)
    place(outs, 2)

    plt.figure(figsize=(6, 4))
    # connections
    for c in genome.conns.values():
        if not c.enabled: continue
        x1, y1 = xs[c.in_id], ys[c.in_id]
        x2, y2 = xs[c.out_id], ys[c.out_id]
        plt.plot([x1, x2], [y1, y2], alpha=0.6)
    # nodes
    for nid in xs:
        n = genome.nodes[nid]
        color = {'in':'lightgray', 'out':'lightgreen', 'hid':'skyblue'}[n.type]
        plt.scatter([xs[nid]], [ys[nid]], s=400, edgecolors='k')
        plt.text(xs[nid], ys[nid], f"{nid}\n{n.activation}", ha='center', va='center', fontsize=8)
    plt.axis('off')
    if fname:
        plt.tight_layout()
        plt.savefig(fname, dpi=200)
        plt.close()
    else:
        plt.show()
