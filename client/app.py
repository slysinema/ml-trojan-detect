"""
Trojan Detection IDS — Desktop Client (Flet 0.84+)

Requirements: pip install flet scapy requests psutil
Run as root/admin for packet capture.
"""

import threading
import time
import statistics
import math
from collections import defaultdict
from datetime import datetime

import flet as ft
import flet.canvas as cv
import requests
import psutil
from scapy.all import sniff, IP, TCP, UDP, conf

conf.verb = 0

# ═══════════════════════════════════════════════════════════════
# Theme
# ═══════════════════════════════════════════════════════════════

BG = "#0B0E14"
BG_CARD = "#12161F"
BG_SURFACE = "#181D2A"
BG_INPUT = "#1A1F2E"
BORDER = "#252B3B"
TEXT = "#E2E8F0"
TEXT_DIM = "#7B8AA0"
TEXT_MUTED = "#4A5568"
ACCENT = "#3B82F6"
GREEN = "#22C55E"
RED = "#EF4444"
YELLOW = "#EAB308"
ORANGE = "#F97316"
CYAN = "#06B6D4"
PURPLE = "#A855F7"

MODEL_COLORS = {"random_forest": ACCENT, "lightgbm": GREEN, "xgboost": ORANGE,
                "extra_trees": PURPLE, "knn": CYAN}
MODEL_NAMES = {"random_forest": "Random Forest", "lightgbm": "LightGBM",
               "xgboost": "XGBoost", "extra_trees": "Extra Trees", "knn": "KNN"}
FEATURE_NAMES = [
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Total Fwd Packets", "Total Backward Packets",
    "Fwd Packets Length Total", "Bwd Packets Length Total",
    "Packet Length Min", "Packet Length Max", "Packet Length Mean",
    "Packet Length Std", "Packet Length Variance",
    "Flow Bytes/s", "Flow Packets/s",
]


# ═══════════════════════════════════════════════════════════════
# Flow Tracker
# ═══════════════════════════════════════════════════════════════

class FlowTracker:
    def __init__(self):
        self.flows = defaultdict(lambda: {
            "timestamps": [], "fwd_packets": 0, "bwd_packets": 0,
            "fwd_bytes": 0, "bwd_bytes": 0, "packet_sizes": [],
            "src_ip": None, "start_time": None,
        })

    def add_packet(self, pkt):
        if not pkt.haslayer(IP):
            return
        ip = pkt[IP]
        sp = pkt[TCP].sport if pkt.haslayer(TCP) else (pkt[UDP].sport if pkt.haslayer(UDP) else 0)
        dp = pkt[TCP].dport if pkt.haslayer(TCP) else (pkt[UDP].dport if pkt.haslayer(UDP) else 0)
        key = tuple(sorted([(ip.src, sp), (ip.dst, dp)])) + (ip.proto,)
        f = self.flows[key]
        now = time.time()
        if f["src_ip"] is None:
            f["src_ip"] = ip.src
            f["start_time"] = now
        f["timestamps"].append(now)
        sz = len(pkt)
        f["packet_sizes"].append(sz)
        if ip.src == f["src_ip"]:
            f["fwd_packets"] += 1; f["fwd_bytes"] += sz
        else:
            f["bwd_packets"] += 1; f["bwd_bytes"] += sz

    def extract_features(self, key):
        f = self.flows.get(key)
        if not f or len(f["timestamps"]) < 3:
            return None
        ts, sz = f["timestamps"], f["packet_sizes"]
        iats = [ts[i] - ts[i - 1] for i in range(1, len(ts))]
        if not iats:
            return None
        dur = ts[-1] - ts[0]
        tb = float(f["fwd_bytes"] + f["bwd_bytes"])
        return [
            statistics.mean(iats), statistics.stdev(iats) if len(iats) > 1 else 0.0,
            max(iats), min(iats),
            float(f["fwd_packets"]), float(f["bwd_packets"]),
            float(f["fwd_bytes"]), float(f["bwd_bytes"]),
            float(min(sz)), float(max(sz)), statistics.mean(sz),
            statistics.stdev(sz) if len(sz) > 1 else 0.0,
            statistics.variance(sz) if len(sz) > 1 else 0.0,
            tb / dur if dur > 0 else 0.0, len(ts) / dur if dur > 0 else 0.0,
        ]

    def clear(self):
        self.flows.clear()


# ═══════════════════════════════════════════════════════════════
# Canvas Charts
# ═══════════════════════════════════════════════════════════════

def make_pie(benign, trojan):
    w, h, cx, cy, r, ir = 200, 200, 100, 100, 75, 35
    total = benign + trojan
    shapes = []
    if total == 0:
        shapes.append(cv.Circle(cx, cy, r, paint=ft.Paint(color=BORDER, style=ft.PaintingStyle.FILL)))
        shapes.append(cv.Circle(cx, cy, ir, paint=ft.Paint(color=BG_CARD, style=ft.PaintingStyle.FILL)))
        shapes.append(cv.Text(cx - 25, cy - 6, "No data", style=ft.TextStyle(color=TEXT_MUTED, size=11)))
    else:
        shapes.append(cv.Circle(cx, cy, r, paint=ft.Paint(color=GREEN, style=ft.PaintingStyle.FILL)))
        if trojan > 0:
            ratio = trojan / total
            shapes.append(cv.Arc(cx - r, cy - r, r * 2, r * 2, -math.pi / 2, ratio * 2 * math.pi,
                                 use_center=True, paint=ft.Paint(color=RED, style=ft.PaintingStyle.FILL)))
        shapes.append(cv.Circle(cx, cy, ir, paint=ft.Paint(color=BG_CARD, style=ft.PaintingStyle.FILL)))
        shapes.append(cv.Text(cx - 18, cy - 6, f"{trojan}/{total}",
                              style=ft.TextStyle(color=TEXT, size=12, weight=ft.FontWeight.BOLD)))
    return cv.Canvas(shapes=shapes, width=w, height=h)


def make_line(data_points):
    w, h = 480, 150
    pl, pr, pt, pb = 45, 10, 10, 22
    cw, ch = w - pl - pr, h - pt - pb
    shapes = []
    if len(data_points) < 2:
        shapes.append(cv.Text(w // 2 - 50, h // 2 - 6, "Waiting for data...",
                              style=ft.TextStyle(color=TEXT_MUTED, size=11)))
        return cv.Canvas(shapes=shapes, width=w, height=h)
    vals = [v for _, v in data_points]
    mx = max(vals) if max(vals) > 0 else 1
    for i in range(5):
        gy = pt + ch - ch * i / 4
        shapes.append(cv.Line(pl, gy, w - pr, gy, paint=ft.Paint(color=BORDER, stroke_width=1)))
        shapes.append(cv.Text(2, gy - 5, str(int(mx * i / 4)),
                              style=ft.TextStyle(color=TEXT_MUTED, size=8)))
    pts = []
    for i, (_, v) in enumerate(data_points):
        x = pl + cw * i / (len(data_points) - 1)
        y = pt + ch - ch * v / mx
        pts.append((x, y))
    for i in range(1, len(pts)):
        shapes.append(cv.Line(pts[i - 1][0], pts[i - 1][1], pts[i][0], pts[i][1],
                              paint=ft.Paint(color=ACCENT, stroke_width=2)))
    for x, y in pts:
        shapes.append(cv.Circle(x, y, 2, paint=ft.Paint(color=ACCENT, style=ft.PaintingStyle.FILL)))
    for idx in [0, len(data_points) // 2, len(data_points) - 1]:
        lbl = data_points[idx][0]
        x = pl + cw * idx / (len(data_points) - 1)
        shapes.append(cv.Text(x - 12, h - 6, lbl, style=ft.TextStyle(color=TEXT_MUTED, size=8)))
    return cv.Canvas(shapes=shapes, width=w, height=h)


def make_bars(probs, thr):
    w, bh = 480, 28
    pl, pr = 110, 50
    cw = w - pl - pr
    n = len(probs)
    if n == 0:
        shapes = [cv.Text(w // 2 - 40, 30, "No data yet", style=ft.TextStyle(color=TEXT_MUTED, size=11))]
        return cv.Canvas(shapes=shapes, width=w, height=80)
    th = n * bh + 18
    shapes = []
    for i, (k, p) in enumerate(probs.items()):
        y = 8 + i * bh
        bw = max(2, cw * p)
        c = MODEL_COLORS.get(k, ACCENT)
        shapes.append(cv.Text(4, y + 3, MODEL_NAMES.get(k, k), style=ft.TextStyle(color=TEXT, size=10)))
        shapes.append(cv.Rect(pl, y, cw, 14, border_radius=4,
                              paint=ft.Paint(color=BORDER, style=ft.PaintingStyle.FILL)))
        shapes.append(cv.Rect(pl, y, bw, 14, border_radius=4,
                              paint=ft.Paint(color=c, style=ft.PaintingStyle.FILL)))
        shapes.append(cv.Text(pl + bw + 5, y + 2, f"{p * 100:.1f}%",
                              style=ft.TextStyle(color=c, size=10, weight=ft.FontWeight.BOLD)))
    tx = pl + cw * thr
    shapes.append(cv.Line(tx, 0, tx, th - 10,
                          paint=ft.Paint(color=RED, stroke_width=1.5, style=ft.PaintingStyle.STROKE)))
    shapes.append(cv.Text(tx - 14, th - 8, f"θ={thr}", style=ft.TextStyle(color=RED, size=8)))
    return cv.Canvas(shapes=shapes, width=w, height=th + 4)


# ═══════════════════════════════════════════════════════════════
# Application
# ═══════════════════════════════════════════════════════════════

def main(page: ft.Page):
    page.title = "Trojan Detection IDS"
    page.bgcolor = BG
    page.padding = 0
    page.window.width = 1200
    page.window.height = 850
    page.theme_mode = ft.ThemeMode.DARK

    is_sniffing = False
    flow_tracker = FlowTracker()
    captured_count = 0
    packet_timeline = []
    analysis_results = []
    benign_count = 0
    trojan_count = 0
    last_model_probs = {}
    server_url = "http://127.0.0.1:8000"
    threshold = 0.90
    min_packets = 10
    selected_models = set(MODEL_NAMES.keys())

    def make_card(content, **kw):
        return ft.Container(content=content, bgcolor=BG_CARD,
                            border=ft.Border.all(1, BORDER), border_radius=12, padding=20, **kw)

    def stitle(t):
        return ft.Text(t, color=TEXT, size=15, weight=ft.FontWeight.W_600)

    # Live counters
    packets_val = ft.Text("0", color=ACCENT, size=22, weight=ft.FontWeight.BOLD)
    flows_val = ft.Text("0", color=CYAN, size=22, weight=ft.FontWeight.BOLD)
    benign_val = ft.Text("0", color=GREEN, size=22, weight=ft.FontWeight.BOLD)
    trojan_val = ft.Text("0", color=RED, size=22, weight=ft.FontWeight.BOLD)
    status_dot = ft.Container(width=10, height=10, border_radius=5, bgcolor=TEXT_MUTED)
    status_lbl = ft.Text("Idle", color=TEXT_DIM, size=12)

    # Chart holders
    pie_h = ft.Container(width=200, height=200)
    line_h = ft.Container(width=490, height=160)
    bar_h = ft.Container(width=490, height=190)

    def refresh_charts():
        pie_h.content = make_pie(benign_count, trojan_count)
        line_h.content = make_line(packet_timeline[-30:])
        bar_h.content = make_bars(last_model_probs, threshold)

    refresh_charts()

    # Log
    log_col = ft.Column(scroll=ft.ScrollMode.AUTO, height=140, spacing=2)

    def add_log(msg, color=TEXT_DIM):
        ts = datetime.now().strftime("%H:%M:%S")
        log_col.controls.append(ft.Text(f"[{ts}] {msg}", color=color, size=11, selectable=True))
        if len(log_col.controls) > 200:
            log_col.controls.pop(0)

    # Results list
    results_list = ft.Column(spacing=6, scroll=ft.ScrollMode.AUTO)

    def build_result_row(res):
        avg = res["avg_prob"]
        threat = avg >= threshold
        clr = RED if threat else GREEN
        verd = "THREAT" if threat else "SAFE"
        ico = ft.Icons.WARNING_ROUNDED if threat else ft.Icons.CHECK_CIRCLE_ROUNDED
        model_items = []
        for mk, mv in res["probs"].items():
            mc = MODEL_COLORS.get(mk, ACCENT)
            model_items.append(ft.Column(spacing=1, controls=[
                ft.Row(alignment=ft.MainAxisAlignment.SPACE_BETWEEN, controls=[
                    ft.Text(MODEL_NAMES.get(mk, mk), color=TEXT_DIM, size=10),
                    ft.Text(f"{mv * 100:.1f}%", color=mc, size=10, weight=ft.FontWeight.BOLD),
                ]),
                ft.ProgressBar(value=mv, height=4, color=mc, bgcolor=BORDER),
            ]))
        feat_items = [ft.Row(alignment=ft.MainAxisAlignment.SPACE_BETWEEN, controls=[
            ft.Text(n, color=TEXT_MUTED, size=10), ft.Text(f"{v:.4f}", color=TEXT_DIM, size=10),
        ]) for n, v in zip(FEATURE_NAMES, res["features"])]
        return ft.Container(
            content=ft.Column(spacing=8, controls=[
                ft.Row(alignment=ft.MainAxisAlignment.SPACE_BETWEEN, controls=[
                    ft.Column(spacing=2, controls=[
                        ft.Text(res["flow"], color=TEXT, size=12, weight=ft.FontWeight.W_500,
                                max_lines=1, overflow=ft.TextOverflow.ELLIPSIS),
                        ft.Text(res["time"], color=TEXT_MUTED, size=10),
                    ]),
                    ft.Row(spacing=6, controls=[
                        ft.Text(f"{avg * 100:.1f}%", color=clr, size=16, weight=ft.FontWeight.BOLD),
                        ft.Icon(ico, color=clr, size=20),
                        ft.Text(verd, color=clr, size=12, weight=ft.FontWeight.BOLD),
                    ]),
                ]),
                ft.Column(model_items, spacing=4),
                ft.ExpansionTile(
                    title=ft.Text("Features", color=TEXT_MUTED, size=10),
                    affinity=ft.TileAffinity.LEADING, initially_expanded=False,
                    collapsed_icon_color=TEXT_MUTED, icon_color=ACCENT,
                    controls=[ft.Container(content=ft.Column(feat_items, spacing=1),
                                           padding=ft.padding.only(left=12, right=12, bottom=8))],
                ),
            ]),
            bgcolor=BG_SURFACE, border=ft.Border.all(1, RED if threat else BORDER),
            border_radius=8, padding=12,
        )

    # ═══ DASHBOARD TAB ═══
    dashboard = ft.Container(
        content=ft.Column(spacing=16, scroll=ft.ScrollMode.AUTO, controls=[
            ft.Row(spacing=12, controls=[
                make_card(ft.Column(spacing=4, horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                                    controls=[ft.Icon(ft.Icons.WIFI_TETHERING_ROUNDED, color=ACCENT, size=20),
                                              packets_val, ft.Text("Packets", color=TEXT_DIM, size=11)]), expand=True),
                make_card(ft.Column(spacing=4, horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                                    controls=[ft.Icon(ft.Icons.ACCOUNT_TREE_ROUNDED, color=CYAN, size=20),
                                              flows_val, ft.Text("Flows", color=TEXT_DIM, size=11)]), expand=True),
                make_card(ft.Column(spacing=4, horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                                    controls=[ft.Icon(ft.Icons.VERIFIED_USER_ROUNDED, color=GREEN, size=20),
                                              benign_val, ft.Text("Benign", color=TEXT_DIM, size=11)]), expand=True),
                make_card(ft.Column(spacing=4, horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                                    controls=[ft.Icon(ft.Icons.BUG_REPORT_ROUNDED, color=RED, size=20),
                                              trojan_val, ft.Text("Threats", color=TEXT_DIM, size=11)]), expand=True),
            ]),
            ft.Row(spacing=12, vertical_alignment=ft.CrossAxisAlignment.START, controls=[
                make_card(ft.Column(spacing=6, horizontal_alignment=ft.CrossAxisAlignment.CENTER, controls=[
                    ft.Text("Classification", color=TEXT_DIM, size=11), pie_h,
                    ft.Row(spacing=12, alignment=ft.MainAxisAlignment.CENTER, controls=[
                        ft.Row(spacing=4, controls=[ft.Container(width=8, height=8, border_radius=4, bgcolor=GREEN),
                                                    ft.Text("Benign", color=TEXT_DIM, size=10)]),
                        ft.Row(spacing=4, controls=[ft.Container(width=8, height=8, border_radius=4, bgcolor=RED),
                                                    ft.Text("Trojan", color=TEXT_DIM, size=10)]),
                    ]),
                ]), width=250),
                make_card(ft.Column(spacing=12, controls=[
                    ft.Column(spacing=4, controls=[ft.Text("Packets Over Time", color=TEXT_DIM, size=11), line_h]),
                    ft.Column(spacing=4, controls=[ft.Text("Model Probabilities", color=TEXT_DIM, size=11), bar_h]),
                ]), expand=True),
            ]),
            make_card(ft.Column(spacing=8, controls=[
                stitle("Event Log"),
                ft.Container(content=log_col, bgcolor=BG_SURFACE,
                             border=ft.Border.all(1, BORDER), border_radius=8, padding=10),
            ])),
        ]),
        padding=20, expand=True,
    )

    # ═══ SCANNER TAB ═══
    ifaces = list(psutil.net_if_addrs().keys()) or ["eth0"]
    iface_dd = ft.Dropdown(
        label="Network Interface", width=280, height=50, text_size=13,
        bgcolor=BG_INPUT, border_color=BORDER, focused_border_color=ACCENT,
        color=TEXT, label_style=ft.TextStyle(color=TEXT_DIM, size=11),
        options=[ft.dropdown.Option(i) for i in ifaces], value=ifaces[0] if ifaces else None,
    )

    model_cbs = {}

    def on_model_toggle(e):
        k = e.control.data
        if e.control.value:
            selected_models.add(k)
        else:
            selected_models.discard(k)
        sel_all_cb.value = len(selected_models) == len(MODEL_NAMES)
        page.update()

    def on_sel_all(e):
        if e.control.value:
            selected_models.update(MODEL_NAMES.keys())
        else:
            selected_models.clear()
        for c in model_cbs.values():
            c.value = e.control.value
        page.update()

    sel_all_cb = ft.Checkbox(label="Select All", value=True, active_color=ACCENT, check_color=BG,
                             label_style=ft.TextStyle(color=TEXT, size=12, weight=ft.FontWeight.BOLD),
                             on_change=on_sel_all)
    cb_list = [sel_all_cb]
    for mk, mn in MODEL_NAMES.items():
        cb = ft.Checkbox(label=mn, value=True, data=mk, active_color=MODEL_COLORS[mk], check_color=BG,
                         label_style=ft.TextStyle(color=TEXT_DIM, size=12), on_change=on_model_toggle)
        model_cbs[mk] = cb
        cb_list.append(cb)

    thr_label = ft.Text(f"Detection Threshold: {threshold:.0%}", color=TEXT, size=12)

    def on_thr(e):
        nonlocal threshold
        threshold = round(e.control.value, 2)
        thr_label.value = f"Detection Threshold: {threshold:.0%}"
        page.update()

    thr_slider = ft.Slider(min=0.5, max=0.99, value=threshold, divisions=49,
                           active_color=RED, inactive_color=BORDER, on_change=on_thr, width=400)

    min_pkt_f = ft.TextField(value=str(min_packets), label="Min Packets / Flow", width=150, height=50,
                             text_size=13, bgcolor=BG_INPUT, border_color=BORDER, focused_border_color=ACCENT,
                             color=TEXT, label_style=ft.TextStyle(color=TEXT_DIM, size=11),
                             keyboard_type=ft.KeyboardType.NUMBER)
    scan_st = ft.Text("Ready to scan", color=TEXT_DIM, size=12)

    def pkt_cb(pkt):
        nonlocal captured_count
        captured_count += 1
        flow_tracker.add_packet(pkt)

    def start_scan(e):
        nonlocal is_sniffing, captured_count, min_packets
        if is_sniffing:
            return
        if not selected_models:
            add_log("Select at least one model!", RED); page.update(); return
        try:
            min_packets = int(min_pkt_f.value)
        except ValueError:
            min_packets = 10
        is_sniffing = True; captured_count = 0; flow_tracker.clear()
        status_dot.bgcolor = GREEN; status_lbl.value = "Capturing"
        start_b.disabled = True; stop_b.disabled = False
        scan_st.value = f"Sniffing on {iface_dd.value}..."; scan_st.color = GREEN
        page.update()
        add_log(f"Capture started on {iface_dd.value} (θ={threshold})", ACCENT); page.update()

        def worker():
            try:
                sniff(iface=iface_dd.value, prn=pkt_cb, store=False, stop_filter=lambda _: not is_sniffing)
            except PermissionError:
                add_log("Run as root/admin!", RED)
            except Exception as ex:
                add_log(f"Error: {ex}", RED)

        threading.Thread(target=worker, daemon=True).start()

        def updater():
            while is_sniffing:
                ts = datetime.now().strftime("%H:%M:%S")
                packet_timeline.append((ts, captured_count))
                if len(packet_timeline) > 60:
                    packet_timeline.pop(0)
                packets_val.value = str(captured_count)
                flows_val.value = str(len(flow_tracker.flows))
                refresh_charts()
                try:
                    page.update()
                except Exception:
                    break
                time.sleep(1)

        threading.Thread(target=updater, daemon=True).start()

    def stop_scan(e):
        nonlocal is_sniffing
        is_sniffing = False
        status_dot.bgcolor = YELLOW; status_lbl.value = "Stopped"
        start_b.disabled = False; stop_b.disabled = True
        scan_st.value = f"Stopped. {captured_count} pkts, {len(flow_tracker.flows)} flows."
        scan_st.color = YELLOW; page.update()
        add_log(f"Stopped: {captured_count} pkts, {len(flow_tracker.flows)} flows", YELLOW); page.update()

    def analyze(e):
        nonlocal benign_count, trojan_count, last_model_probs
        url = server_url.rstrip("/")
        analyzed = 0
        results_list.controls.clear()
        add_log(f"Analyzing → {url}/analyze ...", ACCENT); page.update()
        for fk, fd in flow_tracker.flows.items():
            if len(fd["timestamps"]) < min_packets:
                continue
            feats = flow_tracker.extract_features(fk)
            if feats is None:
                continue
            sa, sb, proto = fk[0], fk[1], fk[2]
            pn = {6: "TCP", 17: "UDP"}.get(proto, str(proto))
            fs = f"{sa[0]}:{sa[1]} ↔ {sb[0]}:{sb[1]} ({pn})"
            try:
                resp = requests.post(f"{url}/analyze", json={"features": feats}, timeout=10)
                data = resp.json()
                if data.get("status") != "success":
                    add_log(f"Server error: {fs}", RED); continue
            except requests.exceptions.ConnectionError:
                add_log(f"Cannot connect to {url}", RED); page.update(); return
            except Exception as ex:
                add_log(f"Error: {ex}", RED); continue
            probs = {mk: data["results"].get(mk, {}).get("trojan_probability", 0.0) for mk in selected_models}
            avg = sum(probs.values()) / len(probs) if probs else 0.0
            if avg >= threshold:
                trojan_count += 1
            else:
                benign_count += 1
            last_model_probs = probs
            entry = {"flow": fs, "features": feats, "probs": probs,
                     "avg_prob": avg, "time": datetime.now().strftime("%H:%M:%S")}
            analysis_results.append(entry)
            results_list.controls.insert(0, build_result_row(entry))
            analyzed += 1
        benign_val.value = str(benign_count); trojan_val.value = str(trojan_count)
        refresh_charts()
        add_log(f"Done: {analyzed} flows. Benign={benign_count}, Threats={trojan_count}", GREEN)
        page.update()

    start_b = ft.Button("Start Capture", icon=ft.Icons.PLAY_ARROW_ROUNDED, on_click=start_scan,
                        bgcolor=GREEN, color=BG,
                        style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8)), height=44, width=180)
    stop_b = ft.Button("Stop Capture", icon=ft.Icons.STOP_ROUNDED, on_click=stop_scan,
                       bgcolor=RED, color=BG, disabled=True,
                       style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8)), height=44, width=180)
    analyze_b = ft.Button("Analyze Flows", icon=ft.Icons.ANALYTICS_ROUNDED, on_click=analyze,
                          bgcolor=ACCENT, color=BG,
                          style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8)), height=44, width=180)

    scanner = ft.Container(
        content=ft.Column(spacing=16, scroll=ft.ScrollMode.AUTO, controls=[
            make_card(ft.Column(spacing=12, controls=[stitle("Capture Configuration"),
                                                      ft.Row(spacing=16, controls=[iface_dd, min_pkt_f])])),
            make_card(ft.Column(spacing=10, controls=[stitle("Model Selection"),
                                                      ft.Row(wrap=True, spacing=10, run_spacing=6, controls=cb_list)])),
            make_card(ft.Column(spacing=8, controls=[
                stitle("Detection Threshold"),
                ft.Text("Minimum average confidence to classify a flow as a threat.", color=TEXT_DIM, size=11),
                thr_label, thr_slider,
            ])),
            ft.Row(spacing=12, controls=[start_b, stop_b, analyze_b, scan_st]),
        ]),
        padding=20, expand=True,
    )

    # ═══ RESULTS TAB ═══
    def clear_res(e):
        nonlocal benign_count, trojan_count, last_model_probs
        benign_count = trojan_count = 0; analysis_results.clear(); last_model_probs.clear()
        results_list.controls.clear(); benign_val.value = "0"; trojan_val.value = "0"
        refresh_charts(); add_log("Results cleared.", TEXT_DIM); page.update()

    results = ft.Container(
        content=ft.Column(spacing=12, expand=True, controls=[
            ft.Row(alignment=ft.MainAxisAlignment.SPACE_BETWEEN, controls=[
                stitle("Analysis Results"),
                ft.TextButton("Clear All", icon=ft.Icons.DELETE_OUTLINE_ROUNDED,
                              on_click=clear_res, style=ft.ButtonStyle(color=TEXT_DIM)),
            ]),
            ft.Container(content=results_list, expand=True),
        ]),
        padding=20, expand=True,
    )

    # ═══ SETTINGS TAB ═══
    url_f = ft.TextField(value=server_url, label="API Server URL", width=400, height=50, text_size=13,
                         bgcolor=BG_INPUT, border_color=BORDER, focused_border_color=ACCENT,
                         color=TEXT, label_style=ft.TextStyle(color=TEXT_DIM, size=11))

    def save_set(e):
        nonlocal server_url
        server_url = url_f.value.strip()
        add_log(f"Settings saved. Server: {server_url}", GREEN); page.update()

    def test_con(e):
        u = url_f.value.strip().rstrip("/")
        try:
            r = requests.get(f"{u}/docs", timeout=5)
            add_log(f"Connection OK ({u}) — {r.status_code}", GREEN)
        except Exception as ex:
            add_log(f"Connection FAILED: {ex}", RED)
        page.update()

    settings = ft.Container(
        content=ft.Column(spacing=16, controls=[
            make_card(ft.Column(spacing=12, controls=[
                stitle("Server Connection"), url_f,
                ft.Row(spacing=12, controls=[
                    ft.Button("Save", icon=ft.Icons.SAVE_ROUNDED, on_click=save_set, bgcolor=ACCENT, color=BG,
                              style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8)), height=40),
                    ft.OutlinedButton("Test Connection", icon=ft.Icons.LAN_ROUNDED, on_click=test_con,
                                     style=ft.ButtonStyle(side=ft.BorderSide(1, ACCENT),
                                                          shape=ft.RoundedRectangleBorder(radius=8)), height=40),
                ]),
            ])),
            make_card(ft.Column(spacing=8, controls=[
                stitle("About"),
                ft.Text("Trojan Detection IDS v2.0", color=TEXT, size=13),
                ft.Text("ML-based IDS for identifying Trojan activity in network traffic. "
                        "Uses 5 models: Random Forest, LightGBM, XGBoost, Extra Trees, KNN.",
                        color=TEXT_DIM, size=11),
                ft.Text("Bachelor Thesis Project", color=TEXT_MUTED, size=11),
            ])),
        ]),
        padding=20, expand=True,
    )

    # ═══ NAVIGATION (manual tabs via buttons) ═══
    views = [dashboard, scanner, results, settings]
    content_area = ft.Container(content=dashboard, expand=True)
    nav_buttons = []

    def switch_tab(idx):
        def handler(e):
            content_area.content = views[idx]
            for i, b in enumerate(nav_buttons):
                b.style = ft.ButtonStyle(
                    color=TEXT if i == idx else TEXT_DIM,
                    bgcolor=BG_SURFACE if i == idx else None,
                    shape=ft.RoundedRectangleBorder(radius=8),
                )
            page.update()
        return handler

    tab_defs = [
        ("Dashboard", ft.Icons.DASHBOARD_ROUNDED),
        ("Scanner", ft.Icons.RADAR_ROUNDED),
        ("Results", ft.Icons.LIST_ALT_ROUNDED),
        ("Settings", ft.Icons.SETTINGS_ROUNDED),
    ]
    for i, (label, icon) in enumerate(tab_defs):
        b = ft.TextButton(
            content=ft.Row(spacing=6, controls=[ft.Icon(icon, size=18), ft.Text(label, size=13)]),
            on_click=switch_tab(i),
            style=ft.ButtonStyle(
                color=TEXT if i == 0 else TEXT_DIM,
                bgcolor=BG_SURFACE if i == 0 else None,
                shape=ft.RoundedRectangleBorder(radius=8),
            ),
            height=40,
        )
        nav_buttons.append(b)

    header = ft.Container(
        content=ft.Row(alignment=ft.MainAxisAlignment.SPACE_BETWEEN, controls=[
            ft.Row(spacing=10, controls=[
                ft.Icon(ft.Icons.SECURITY_ROUNDED, color=ACCENT, size=24),
                ft.Text("Trojan Detection IDS", color=TEXT, size=18, weight=ft.FontWeight.BOLD),
            ]),
            ft.Row(spacing=8, controls=[status_dot, status_lbl]),
        ]),
        bgcolor=BG_CARD, border=ft.Border.only(bottom=ft.BorderSide(1, BORDER)),
        padding=ft.padding.symmetric(horizontal=20, vertical=12),
    )

    nav_bar = ft.Container(
        content=ft.Row(spacing=4, controls=nav_buttons),
        bgcolor=BG_CARD, border=ft.Border.only(bottom=ft.BorderSide(1, BORDER)),
        padding=ft.padding.symmetric(horizontal=16, vertical=6),
    )

    page.add(ft.Column(expand=True, spacing=0, controls=[header, nav_bar, content_area]))
    add_log("Application started. Go to Scanner to begin.", ACCENT)
    page.update()


if __name__ == "__main__":
    ft.run(main)