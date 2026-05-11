"""Build a proper Excalidraw JSON export with bound text labels and arrows.
Run: python3 build_excalidraw.py > architecture.excalidraw
"""
import json, random, time

random.seed(42)
NOW = int(time.time() * 1000)

def seed():
    return random.randint(1, 2**31 - 1)

def base(eid, x, y, w, h, **kw):
    """Common Excalidraw element fields."""
    d = {
        "id": eid, "x": x, "y": y, "width": w, "height": h,
        "angle": 0, "strokeColor": "#1e1e1e", "backgroundColor": "transparent",
        "fillStyle": "solid", "strokeWidth": 2, "strokeStyle": "solid",
        "roughness": 1, "opacity": 100,
        "groupIds": [], "frameId": None, "index": None,
        "roundness": None, "seed": seed(), "version": 1,
        "versionNonce": seed(), "isDeleted": False,
        "boundElements": None, "updated": NOW, "link": None, "locked": False,
    }
    d.update(kw)
    return d

def rect(eid, x, y, w, h, fill, stroke, label_id=None, arrows=None):
    bound = []
    if label_id:
        bound.append({"type": "text", "id": label_id})
    if arrows:
        for a in arrows:
            bound.append({"type": "arrow", "id": a})
    return base(eid, x, y, w, h,
        type="rectangle", backgroundColor=fill, strokeColor=stroke,
        roundness={"type": 3},
        boundElements=bound if bound else None)

def label(eid, container_id, text, font_size=14):
    """Bound text inside a rectangle (centered)."""
    return base(eid, 0, 0, 1, 1,
        type="text", text=text, originalText=text,
        fontSize=font_size, fontFamily=1,
        textAlign="center", verticalAlign="middle",
        containerId=container_id,
        lineHeight=1.25)

def stext(eid, x, y, text, font_size=14, color="#1e1e1e"):
    """Standalone text (titles, lane labels)."""
    return base(eid, x, y, max(50, len(text) * font_size // 2), font_size + 6,
        type="text", text=text, originalText=text,
        fontSize=font_size, fontFamily=1,
        textAlign="left", verticalAlign="top",
        strokeColor=color,
        containerId=None,
        lineHeight=1.25)

def arrow(eid, x, y, dx, dy, start_id, end_id, color="#1e1e1e", dashed=False):
    sb = {"elementId": start_id, "focus": 0, "gap": 4}
    eb = {"elementId": end_id, "focus": 0, "gap": 4}
    return base(eid, x, y, abs(dx), abs(dy),
        type="arrow", strokeColor=color,
        strokeStyle="dashed" if dashed else "solid",
        points=[[0, 0], [dx, dy]],
        lastCommittedPoint=None,
        startBinding=sb, endBinding=eb,
        startArrowhead=None, endArrowhead="arrow")

elements = []

# Title and subtitle
elements.append(stext("title", 290, 10, "GNN-ProtoNet Architecture", 26))
elements.append(stext("sub", 230, 50, "Cross-site Parkinson's EEG Detection (Strict LODO, 230 subjects)", 15, "#757575"))

# Lane 1 — DATA
elements.append(stext("l1lab", 20, 110, "DATA", 14, "#868e96"))
elements.append(rect("d1", 90, 85, 260, 80, "#ffd8a8", "#c2410c", "d1t", ["da1"]))
elements.append(label("d1t", "d1", "UC San Diego (ds002778)\n50 subj  (25 PD / 25 HC)\n67 ch  •  512 Hz", 14))
elements.append(rect("d2", 380, 85, 260, 80, "#ffd8a8", "#c2410c", "d2t", ["da2"]))
elements.append(label("d2t", "d2", "UNM (ds003490)\n31 subj  (16 PD / 15 HC)\n64 ch BDF  •  500 Hz", 14))
elements.append(rect("d3", 670, 85, 260, 80, "#ffd8a8", "#c2410c", "d3t", ["da3"]))
elements.append(label("d3t", "d3", "Iowa (ds004584)\n149 subj  (100 PD / 49 HC)\n64 ch  •  500 Hz", 14))
elements.append(stext("converge", 325, 205, "230 subjects total  (141 PD / 89 HC)", 15))

# Lane 2 — PREPROCESS
elements.append(stext("l2lab", 20, 275, "PREPROCESS", 14, "#868e96"))
elements.append(rect("prep", 195, 240, 630, 85, "#dbe4ff", "#1e40af", "prept", ["da1", "da2", "da3", "ap"]))
elements.append(label("prept", "prep", "Resample 500 Hz  •  Bandpass 0.5–50 Hz (FIR)  •  Notch 50/60 Hz\nHarmonize to canonical 32-channel 10-20 montage  (spherical-spline interp.)\nSegment into 1-second non-overlapping epochs", 14))

# Arrows from datasets to preprocess
elements.append(arrow("da1", 220, 165, 290, 75, "d1", "prep", "#c2410c"))
elements.append(arrow("da2", 510, 165, 0, 75, "d2", "prep", "#c2410c"))
elements.append(arrow("da3", 800, 165, -290, 75, "d3", "prep", "#c2410c"))

# Lane 3 — FEATURES (two parallel boxes)
elements.append(stext("l3lab", 20, 405, "FEATURES", 14, "#868e96"))
elements.append(rect("f1", 195, 350, 290, 135, "#c3fae8", "#0e7490", "f1t", ["ap1", "afg1"]))
elements.append(label("f1t", "f1", "Node features (13-dim per node)\n• PSD in 5 bands  (δ θ α β γ)\n• Time-domain (mean, std, skew, kurt)\n• Hjorth (activity, mobility, complex.)\n• Sample entropy", 13))
elements.append(rect("f2", 535, 350, 290, 135, "#c3fae8", "#0e7490", "f2t", ["ap2", "afg2"]))
elements.append(label("f2t", "f2", "Edge weights\n• PLV computed in 5 bands\n• Avg across bands  →  32×32 matrix\n• Top-k = 8 sparsification\n• Undirected, weighted", 13))

# Arrows preprocess to features
elements.append(arrow("ap1", 350, 325, -10, 25, "prep", "f1", "#1e40af"))
elements.append(arrow("ap2", 670, 325, 10, 25, "prep", "f2", "#1e40af"))

# Lane 4 — GRAPH
elements.append(stext("l4lab", 20, 545, "GRAPH", 14, "#868e96"))
elements.append(rect("g", 280, 520, 460, 75, "#eebefa", "#a21caf", "gt", ["afg1", "afg2", "agat", "agcn"]))
elements.append(label("gt", "g", "32-node EEG graph per epoch\n32 nodes × 13 features  •  ~250 weighted edges", 14))

# Arrows features to graph
elements.append(arrow("afg1", 340, 485, 110, 35, "f1", "g", "#0e7490"))
elements.append(arrow("afg2", 680, 485, -110, 35, "f2", "g", "#0e7490"))

# Lane 5 — ENCODERS
elements.append(stext("l5lab", 20, 680, "ENCODERS", 14, "#868e96"))
elements.append(rect("gat", 195, 640, 290, 120, "#a5d8ff", "#1971c2", "gatt", ["agat", "agatr", "agate"]))
elements.append(label("gatt", "gat", "GAT encoder\n3 GATConv layers\nheads (4, 4, 1)  •  hidden 64/head\nELU + BatchNorm  •  dropout 0.3\nedge_dim = 1  (PLV → attention bias)", 13))
elements.append(rect("gcn", 535, 640, 290, 120, "#b2f2bb", "#15803d", "gcnt", ["agcn", "agcnr", "agcne"]))
elements.append(label("gcnt", "gcn", "GCN encoder\n3 GCNConv layers\nhidden 256\nELU + BatchNorm  •  dropout 0.3\nPLV as edge_weight", 13))

# Arrows graph to encoders
elements.append(arrow("agat", 420, 595, -80, 45, "g", "gat", "#a21caf"))
elements.append(arrow("agcn", 600, 595, 80, 45, "g", "gcn", "#a21caf"))

# Lane 6 — READOUT
elements.append(stext("l6lab", 20, 810, "READOUT", 14, "#868e96"))
elements.append(rect("read", 215, 785, 590, 75, "#fff3bf", "#a16207", "readt", ["agatr", "agcnr", "areadp"]))
elements.append(label("readt", "read", "Dual readout: mean pool + max pool over 32 nodes  →  concat (256-dim)\nMLP 256 → 128  →  graph embedding", 14))

# Arrows encoders to readout
elements.append(arrow("agatr", 340, 760, 80, 25, "gat", "read", "#1971c2"))
elements.append(arrow("agcnr", 680, 760, -80, 25, "gcn", "read", "#15803d"))

# Lane 7 — PROTOTYPICAL
elements.append(stext("l7lab", 20, 920, "PROTOTYPICAL", 14, "#868e96"))
elements.append(rect("proto", 215, 885, 590, 95, "#d0bfff", "#6d28d9", "protot", ["areadp", "aprotod"]))
elements.append(label("protot", "proto", "Class prototypes from K = 5 support graphs per class\nPrototype = mean of support embeddings\nQuery classified by argmin Euclidean distance", 14))
elements.append(arrow("areadp", 510, 860, 0, 25, "read", "proto", "#a16207"))

# Lane 8 — DECISION
elements.append(stext("l8lab", 20, 1030, "DECISION", 14, "#868e96"))
elements.append(rect("dec", 260, 1005, 500, 75, "#d0bfff", "#6d28d9", "dect", ["aprotod", "adecres"]))
elements.append(label("dect", "dec", "Subject-level decision\nMajority vote across all query epochs  →  PD or HC", 14))
elements.append(arrow("aprotod", 510, 980, 0, 25, "proto", "dec", "#6d28d9"))

# ENSEMBLE side branch
elements.append(stext("enslab", 850, 640, "ENSEMBLE", 14, "#868e96"))
elements.append(rect("ens", 850, 665, 230, 120, "#ffc9c9", "#b91c1c", "enst", ["agate", "agcne", "aensres"]))
elements.append(label("enst", "ens", "Late-fusion ensemble\nAverage GAT + GCN\nclass-1 probabilities\nper epoch\n→ majority vote\nper subject", 12))
elements.append(arrow("agate", 485, 680, 365, 25, "gat", "ens", "#1971c2", dashed=True))
elements.append(arrow("agcne", 825, 720, 25, 5, "gcn", "ens", "#15803d", dashed=True))

# Lane 9 — RESULTS
elements.append(stext("l9lab", 20, 1170, "RESULTS", 14, "#868e96"))
elements.append(rect("res", 120, 1115, 900, 160, "#ffc9c9", "#b91c1c", "rest", ["adecres", "aensres"]))
elements.append(label("rest", "res", "Strict Leave-One-Dataset-Out  •  230 subjects  •  K = 5\n\nGCN single:    94.47% accuracy   •   0.9845 AUC   •   217 / 230 subjects correct\nEnsemble:       98.22% accuracy   •   0.9961 AUC   •   226 / 230 subjects correct\n\nBeats prior cross-site SOTA:  TransformEEG 80.10%  •  ARP-N 80.6%  •  MCPNet 90.2% (2 datasets only)", 13))
elements.append(arrow("adecres", 510, 1080, 60, 35, "dec", "res", "#6d28d9"))
elements.append(arrow("aensres", 965, 785, -360, 330, "ens", "res", "#b91c1c", dashed=True))

doc = {
    "type": "excalidraw",
    "version": 2,
    "source": "https://excalidraw.com",
    "elements": elements,
    "appState": {
        "viewBackgroundColor": "#ffffff",
        "gridSize": None,
    },
    "files": {},
}

print(json.dumps(doc, indent=2))
