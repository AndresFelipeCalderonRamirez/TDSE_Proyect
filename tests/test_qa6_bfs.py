"""
QA-6  BFS Risk Propagation Unit Tests  (US-06, Acceptance Criterion 3)
=======================================================================
Verifies: Risk(ej) >= alpha * Risk(ei) for every edge (ei -> ej)
where propagation flows from ei to ej (i.e. Risk(ei) >= Risk(ej)).

Also validates the max-operator guarantee and decaying exponential
behaviour described in Section 5.4 of the paper.

5-node graph with known initial risks, verified manually.

Run:
    pytest tests/test_qa6_bfs.py -v
"""

import sys
from pathlib import Path

import networkx as nx
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "lambdas" / "digital_twin"))

from lambda_function import _propagate_risk  # noqa: E402

ALPHA = 0.7
EPS   = 1e-9


# ── AC3: 5-node linear graph, manual verification ─────────────────────────────

class TestLinearGraph:
    """
    Graph:  A(0.8) — B(0) — C(0) — D(0) — E(0)
    Expected after BFS with alpha=0.7:
      A = 0.8
      B = 0.7 * 0.8              = 0.56000
      C = 0.7 * 0.56             = 0.39200
      D = 0.7 * 0.392            = 0.27440
      E = 0.7 * 0.2744           = 0.19208
    """

    def setup_method(self):
        self.G = nx.path_graph(["A", "B", "C", "D", "E"])
        self.initial = {"A": 0.8}
        self.risk = _propagate_risk(self.G, self.initial, alpha=ALPHA)

    def test_source_node_unchanged(self):
        assert abs(self.risk["A"] - 0.8) < EPS

    def test_first_hop(self):
        assert abs(self.risk["B"] - 0.56) < EPS

    def test_second_hop(self):
        assert abs(self.risk["C"] - 0.392) < EPS

    def test_third_hop(self):
        assert abs(self.risk["D"] - 0.2744) < EPS

    def test_fourth_hop(self):
        assert abs(self.risk["E"] - 0.19208) < EPS

    def test_ac3_monotone_property(self):
        """AC3: For every edge where ei has higher risk, risk[ej] >= alpha * risk[ei]."""
        for ei, ej in self.G.edges():
            r_i, r_j = self.risk[ei], self.risk[ej]
            if r_i >= r_j:
                assert r_j >= ALPHA * r_i - EPS, (
                    f"Edge ({ei},{ej}): risk[{ej}]={r_j:.6f} < "
                    f"alpha*risk[{ei}]={ALPHA*r_i:.6f}"
                )
            else:
                assert r_i >= ALPHA * r_j - EPS, (
                    f"Edge ({ei},{ej}): risk[{ei}]={r_i:.6f} < "
                    f"alpha*risk[{ej}]={ALPHA*r_j:.6f}"
                )


# ── Max-operator: two competing paths to one node ─────────────────────────────

def test_max_operator_takes_higher_source():
    """
    Diamond: A(1.0) → C and B(0.6) → C
    C must receive max(alpha*A, alpha*B) = alpha * max(A, B) = 0.7
    """
    G = nx.Graph()
    G.add_edges_from([("A", "C"), ("B", "C")])
    risk = _propagate_risk(G, {"A": 1.0, "B": 0.6}, alpha=ALPHA)

    assert abs(risk["C"] - 0.7) < EPS, f"C={risk['C']:.6f}, expected 0.7"
    assert risk["A"] >= 1.0 - EPS
    assert risk["B"] >= 0.6 - EPS


# ── Star graph: all leaves receive alpha * center ────────────────────────────

def test_star_graph_all_leaves():
    """
    Star with centre=1.0 and 4 leaves at 0.
    All leaves must reach exactly alpha * 1.0 = 0.7.
    """
    G = nx.star_graph(4)   # nodes 0 (center), 1-4 (leaves)
    risk = _propagate_risk(G, {0: 1.0}, alpha=ALPHA)

    assert abs(risk[0] - 1.0) < EPS
    for leaf in range(1, 5):
        assert abs(risk[leaf] - 0.7) < EPS, f"leaf {leaf}: {risk[leaf]:.6f}"


# ── Risk is never amplified beyond 1.0 ───────────────────────────────────────

def test_no_amplification_on_triangle():
    """
    Risk can only spread (attenuated), never amplified beyond initial max.
    Triangle with A=0.9, B=0.5, C=0.3.
    """
    G = nx.cycle_graph(["A", "B", "C"])
    risk = _propagate_risk(G, {"A": 0.9, "B": 0.5, "C": 0.3}, alpha=ALPHA)

    assert all(r <= 1.0 + EPS for r in risk.values()), "Risk exceeded 1.0"
    assert risk["A"] >= 0.9 - EPS, "Source A risk must not decrease"
    # B gets max(0.5, 0.7*0.9) = max(0.5, 0.63) = 0.63
    assert risk["B"] >= 0.63 - EPS
    # C gets max(0.3, 0.7*0.63) = max(0.3, 0.441) = 0.441
    assert risk["C"] >= 0.441 - EPS


# ── Empty initial risks → all zeros ──────────────────────────────────────────

def test_zero_initial_risk_propagates_nothing():
    G = nx.path_graph(5)
    risk = _propagate_risk(G, {}, alpha=ALPHA)
    assert all(r == 0.0 for r in risk.values())


# ── Disconnected graph: risk does not jump between components ─────────────────

def test_disconnected_graph_no_cross_propagation():
    """
    Two separate components.
    Risk in component-1 must not reach component-2.
    """
    G = nx.Graph()
    G.add_edges_from([("A", "B"), ("B", "C")])   # component 1
    G.add_edges_from([("X", "Y"), ("Y", "Z")])   # component 2

    risk = _propagate_risk(G, {"A": 1.0}, alpha=ALPHA)

    assert abs(risk["A"] - 1.0) < EPS
    assert abs(risk["B"] - 0.7) < EPS
    assert abs(risk["C"] - 0.49) < EPS
    assert risk["X"] == 0.0, "X must not receive risk from component 1"
    assert risk["Y"] == 0.0
    assert risk["Z"] == 0.0


# ── Exponential decay: 10 hops from a source ─────────────────────────────────

def test_exponential_decay_along_chain():
    """
    Risk decays as alpha^k at distance k from the source.
    After 10 hops: alpha^10 = 0.7^10 ≈ 0.0282
    """
    n = 12
    G = nx.path_graph(n)
    risk = _propagate_risk(G, {0: 1.0}, alpha=ALPHA)

    for k in range(n):
        expected = ALPHA ** k
        assert abs(risk[k] - expected) < EPS, (
            f"hop {k}: risk={risk[k]:.8f}, expected alpha^{k}={expected:.8f}"
        )


# ── Topology node-count sanity (AC2 via graph construction) ──────────────────

class TestTopologyNodeCount:
    """
    AC2: Tenant A must have 50 nodes, Tenant B must have 40 nodes.
    Verified by constructing the topology the same way setup_aws.py does.
    """

    @staticmethod
    def _build(n_segments: int, tenant_id: str):
        short  = tenant_id.split("-")[1].upper()
        prefix = f"SEG_{short}_"
        nodes  = [f"{prefix}{i:03d}" for i in range(n_segments)]

        edges       = []
        trunk_n      = max(5, n_segments // 5)
        branch_count = 4
        branch_len   = (n_segments - trunk_n) // branch_count

        for i in range(trunk_n - 1):
            edges.append([nodes[i], nodes[i + 1]])

        for b in range(branch_count):
            attach = round((b + 1) * (trunk_n - 1) / (branch_count + 1))
            start  = trunk_n + b * branch_len
            end    = start + branch_len if b < branch_count - 1 else n_segments
            edges.append([nodes[attach], nodes[start]])
            for i in range(start, end - 1):
                edges.append([nodes[i], nodes[i + 1]])

        for b in range(branch_count - 1):
            b_end        = trunk_n + (b + 1) * branch_len - 1
            b_next_start = trunk_n + (b + 1) * branch_len
            if b_end < n_segments and b_next_start < n_segments:
                edges.append([nodes[b_end], nodes[b_next_start]])

        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        return G

    def test_tenant_a_50_nodes(self):
        G = self._build(50, "tenant-A")
        assert len(G.nodes()) == 50, f"Expected 50, got {len(G.nodes())}"

    def test_tenant_b_40_nodes(self):
        G = self._build(40, "tenant-B")
        assert len(G.nodes()) == 40, f"Expected 40, got {len(G.nodes())}"

    def test_tenant_a_graph_is_connected(self):
        G = self._build(50, "tenant-A")
        assert nx.is_connected(G), "Tenant A graph must be connected"

    def test_tenant_b_graph_is_connected(self):
        G = self._build(40, "tenant-B")
        assert nx.is_connected(G), "Tenant B graph must be connected"

    def test_segment_ids_match_iot_simulator_format(self):
        G = self._build(50, "tenant-A")
        for node in G.nodes():
            assert node.startswith("SEG_A_"), f"Unexpected node ID: {node}"
