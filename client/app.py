"""
Trojan Detection IDS — Desktop Client (Flet)

A dark-themed desktop application that captures live network traffic,
extracts 15 flow-level features from packets, sends them to the FastAPI
server for analysis, and displays Trojan probability from 5 ML models.

Features extracted per flow:
    Flow IAT Mean, Flow IAT Std, Flow IAT Max, Flow IAT Min,
    Total Fwd Packets, Total Backward Packets,
    Fwd Packets Length Total, Bwd Packets Length Total,
    Packet Length Min, Packet Length Max, Packet Length Mean,
    Packet Length Std, Packet Length Variance,
    Flow Bytes/s, Flow Packets/s

Requirements:
    pip install flet scapy requests psutil
    (Run as Administrator / root for packet capture)
"""

import threading
import time
import statistics
from collections import defaultdict
from datetime import datetime

import flet as ft
import requests
import psutil
from scapy.all import sniff, IP, TCP, UDP, conf

# Disable verbose scapy output
conf.verb = 0

# ──────────────────────────────────────────────────────────────────
# Flow tracker — accumulates per-flow packet data
# ──────────────────────────────────────────────────────────────────

class FlowTracker:
    """
    Tracks network flows identified by 5-tuple (src_ip, dst_ip, src_port, dst_port, proto).
    Accumulates packet timestamps and sizes to compute 15 features.
    """

    def __init__(self):
        self.flows: dict[tuple, dict] = defaultdict(lambda: {
            "timestamps": [],
            "fwd_packets": 0,
            "bwd_packets": 0,
            "fwd_bytes": 0,
            "bwd_bytes": 0,
            "packet_sizes": [],
            "src_ip": None,
            "start_time": None,
        })

    def add_packet(self, pkt):
        """Register a captured packet into its corresponding flow."""
        if not pkt.haslayer(IP):
            return None

        ip_layer = pkt[IP]
        src_ip = ip_layer.src
        dst_ip = ip_layer.dst
        proto = ip_layer.proto

        src_port = 0
        dst_port = 0
        if pkt.haslayer(TCP):
            src_port = pkt[TCP].sport
            dst_port = pkt[TCP].dport
        elif pkt.haslayer(UDP):
            src_port = pkt[UDP].sport
            dst_port = pkt[UDP].dport

        # Canonical flow key (sorted so both directions map to same flow)
        flow_key = tuple(sorted([(src_ip, src_port), (dst_ip, dst_port)])) + (proto,)

        flow = self.flows[flow_key]
        now = time.time()

        if flow["src_ip"] is None:
            flow["src_ip"] = src_ip
            flow["start_time"] = now

        flow["timestamps"].append(now)
        pkt_len = len(pkt)
        flow["packet_sizes"].append(pkt_len)

        # Determine direction (forward = same as first seen src)
        if src_ip == flow["src_ip"]:
            flow["fwd_packets"] += 1
            flow["fwd_bytes"] += pkt_len
        else:
            flow["bwd_packets"] += 1
            flow["bwd_bytes"] += pkt_len

        return flow_key

    def extract_features(self, flow_key) -> list[float] | None:
        """
        Compute the 15 features for a given flow.
        Returns None if there are not enough packets.
        """
        flow = self.flows.get(flow_key)
        if flow is None or len(flow["timestamps"]) < 3:
            return None

        timestamps = flow["timestamps"]
        sizes = flow["packet_sizes"]

        # Inter-Arrival Times
        iats = [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]

        if len(iats) == 0:
            return None

        flow_iat_mean = statistics.mean(iats)
        flow_iat_std = statistics.stdev(iats) if len(iats) > 1 else 0.0
        flow_iat_max = max(iats)
        flow_iat_min = min(iats)

        total_fwd_packets = float(flow["fwd_packets"])
        total_bwd_packets = float(flow["bwd_packets"])
        fwd_bytes_total = float(flow["fwd_bytes"])
        bwd_bytes_total = float(flow["bwd_bytes"])

        pkt_len_min = float(min(sizes))
        pkt_len_max = float(max(sizes))
        pkt_len_mean = statistics.mean(sizes)
        pkt_len_std = statistics.stdev(sizes) if len(sizes) > 1 else 0.0
        pkt_len_variance = statistics.variance(sizes) if len(sizes) > 1 else 0.0

        duration = timestamps[-1] - timestamps[0]
        if duration > 0:
            flow_bytes_per_sec = (fwd_bytes_total + bwd_bytes_total) / duration
            flow_packets_per_sec = len(timestamps) / duration
        else:
            flow_bytes_per_sec = 0.0
            flow_packets_per_sec = 0.0

        return [
            flow_iat_mean, flow_iat_std, flow_iat_max, flow_iat_min,
            total_fwd_packets, total_bwd_packets,
            fwd_bytes_total, bwd_bytes_total,
            pkt_len_min, pkt_len_max, pkt_len_mean,
            pkt_len_std, pkt_len_variance,
            flow_bytes_per_sec, flow_packets_per_sec
        ]

    def clear(self):
        """Reset all tracked flows."""
        self.flows.clear()


# ──────────────────────────────────────────────────────────────────
# Feature names for display
# ──────────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Total Fwd Packets", "Total Backward Packets",
    "Fwd Packets Length Total", "Bwd Packets Length Total",
    "Packet Length Min", "Packet Length Max", "Packet Length Mean",
    "Packet Length Std", "Packet Length Variance",
    "Flow Bytes/s", "Flow Packets/s"
]

MODEL_DISPLAY_NAMES = {
    "random_forest": "Random Forest",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
    "extra_trees": "Extra Trees",
    "knn": "KNN"
}

# ──────────────────────────────────────────────────────────────────
# Color palette
# ──────────────────────────────────────────────────────────────────

BG_PRIMARY = "#0D1117"
BG_CARD = "#161B22"
BG_INPUT = "#1C2128"
BORDER_COLOR = "#30363D"
TEXT_PRIMARY = "#E6EDF3"
TEXT_SECONDARY = "#8B949E"
ACCENT_BLUE = "#58A6FF"
ACCENT_GREEN = "#3FB950"
ACCENT_RED = "#F85149"
ACCENT_YELLOW = "#D29922"
ACCENT_ORANGE = "#DB6D28"


# ──────────────────────────────────────────────────────────────────
# Main application
# ──────────────────────────────────────────────────────────────────

def main(page: ft.Page):
    page.title = "Trojan Detection IDS"
    page.bgcolor = BG_PRIMARY
    page.padding = 20
    page.window.width = 1100
    page.window.height = 820
    page.theme_mode = ft.ThemeMode.DARK

    # ── State ──
    sniffer_thread = None
    is_sniffing = False
    flow_tracker = FlowTracker()
    captured_count = 0

    # ── Server URL input ──
    server_url_field = ft.TextField(
        value="http://127.0.0.1:8000",
        label="API Server URL",
        width=350,
        height=48,
        text_size=13,
        bgcolor=BG_INPUT,
        border_color=BORDER_COLOR,
        focused_border_color=ACCENT_BLUE,
        color=TEXT_PRIMARY,
        label_style=ft.TextStyle(color=TEXT_SECONDARY, size=12),
    )

    # ── Network interface dropdown ──
    ifaces = []
    try:
        net_if = psutil.net_if_addrs()
        ifaces = list(net_if.keys())
    except Exception:
        ifaces = ["eth0", "wlan0"]

    iface_dropdown = ft.Dropdown(
        label="Network Interface",
        width=250,
        height=48,
        text_size=13,
        bgcolor=BG_INPUT,
        border_color=BORDER_COLOR,
        focused_border_color=ACCENT_BLUE,
        color=TEXT_PRIMARY,
        label_style=ft.TextStyle(color=TEXT_SECONDARY, size=12),
        options=[ft.dropdown.Option(i) for i in ifaces],
        value=ifaces[0] if ifaces else None,
    )

    # ── Min packets per flow ──
    min_packets_field = ft.TextField(
        value="10",
        label="Min packets per flow",
        width=160,
        height=48,
        text_size=13,
        bgcolor=BG_INPUT,
        border_color=BORDER_COLOR,
        focused_border_color=ACCENT_BLUE,
        color=TEXT_PRIMARY,
        label_style=ft.TextStyle(color=TEXT_SECONDARY, size=12),
        keyboard_type=ft.KeyboardType.NUMBER,
    )

    # ── Status indicators ──
    status_text = ft.Text("Status: Idle", color=TEXT_SECONDARY, size=13)
    packets_counter = ft.Text("Packets: 0", color=TEXT_SECONDARY, size=13)
    flows_counter = ft.Text("Flows: 0", color=TEXT_SECONDARY, size=13)

    # ── Log area ──
    log_column = ft.Column(
        scroll=ft.ScrollMode.AUTO,
        height=180,
        spacing=4,
    )

    def add_log(message: str, color: str = TEXT_SECONDARY):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_column.controls.append(
            ft.Text(f"[{timestamp}] {message}", color=color, size=12, selectable=True)
        )
        # Keep only last 100 entries
        if len(log_column.controls) > 100:
            log_column.controls.pop(0)
        page.update()

    # ── Results area ──
    results_column = ft.Column(spacing=8)

    def build_result_card(flow_key_str: str, features: list[float], server_response: dict):
        """Build a visual card showing analysis results for one flow."""
        results = server_response.get("results", {})

        # Model probability bars
        model_rows = []
        for model_key, display_name in MODEL_DISPLAY_NAMES.items():
            prob = results.get(model_key, {}).get("trojan_probability", 0.0)
            prob_pct = prob * 100

            if prob < 0.3:
                bar_color = ACCENT_GREEN
                label_color = ACCENT_GREEN
            elif prob < 0.7:
                bar_color = ACCENT_YELLOW
                label_color = ACCENT_YELLOW
            else:
                bar_color = ACCENT_RED
                label_color = ACCENT_RED

            model_rows.append(
                ft.Column(
                    spacing=2,
                    controls=[
                        ft.Row(
                            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                            controls=[
                                ft.Text(display_name, color=TEXT_PRIMARY, size=12, weight=ft.FontWeight.W_500),
                                ft.Text(f"{prob_pct:.2f}%", color=label_color, size=12, weight=ft.FontWeight.BOLD),
                            ]
                        ),
                        ft.ProgressBar(
                            value=prob,
                            width=450,
                            height=6,
                            color=bar_color,
                            bgcolor=BG_INPUT,
                        ),
                    ]
                )
            )

        # Average probability
        probs = [results.get(k, {}).get("trojan_probability", 0.0) for k in MODEL_DISPLAY_NAMES]
        avg_prob = sum(probs) / len(probs) if probs else 0.0

        if avg_prob < 0.3:
            verdict = "LOW RISK"
            verdict_color = ACCENT_GREEN
            verdict_icon = ft.icons.CHECK_CIRCLE_OUTLINED
        elif avg_prob < 0.7:
            verdict = "MEDIUM RISK"
            verdict_color = ACCENT_YELLOW
            verdict_icon = ft.icons.WARNING_AMBER_ROUNDED
        else:
            verdict = "HIGH RISK"
            verdict_color = ACCENT_RED
            verdict_icon = ft.icons.DANGEROUS_ROUNDED

        # Features expandable section
        feature_rows = []
        for name, val in zip(FEATURE_NAMES, features):
            feature_rows.append(
                ft.Row(
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    controls=[
                        ft.Text(name, color=TEXT_SECONDARY, size=11),
                        ft.Text(f"{val:.6f}", color=TEXT_PRIMARY, size=11),
                    ]
                )
            )

        features_expansion = ft.ExpansionTile(
            title=ft.Text("Extracted Features (15)", color=TEXT_SECONDARY, size=12),
            affinity=ft.TileAffinity.LEADING,
            initially_expanded=False,
            controls=[
                ft.Container(
                    content=ft.Column(feature_rows, spacing=2),
                    padding=ft.padding.only(left=16, right=16, bottom=8),
                )
            ],
            collapsed_icon_color=TEXT_SECONDARY,
            icon_color=ACCENT_BLUE,
        )

        card = ft.Container(
            content=ft.Column(
                spacing=12,
                controls=[
                    # Header: flow key + verdict
                    ft.Row(
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                        controls=[
                            ft.Text(flow_key_str, color=TEXT_PRIMARY, size=12,
                                    weight=ft.FontWeight.W_500, max_lines=1,
                                    overflow=ft.TextOverflow.ELLIPSIS, width=400),
                            ft.Row(
                                spacing=6,
                                controls=[
                                    ft.Icon(verdict_icon, color=verdict_color, size=18),
                                    ft.Text(verdict, color=verdict_color, size=13, weight=ft.FontWeight.BOLD),
                                ]
                            ),
                        ]
                    ),
                    ft.Divider(height=1, color=BORDER_COLOR),
                    # Model results
                    ft.Column(model_rows, spacing=8),
                    ft.Divider(height=1, color=BORDER_COLOR),
                    # Average
                    ft.Row(
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                        controls=[
                            ft.Text("Average Probability", color=TEXT_PRIMARY, size=13, weight=ft.FontWeight.BOLD),
                            ft.Text(f"{avg_prob * 100:.2f}%", color=verdict_color, size=14, weight=ft.FontWeight.BOLD),
                        ]
                    ),
                    # Features
                    features_expansion,
                ]
            ),
            bgcolor=BG_CARD,
            border=ft.border.all(1, BORDER_COLOR),
            border_radius=10,
            padding=16,
            margin=ft.margin.only(bottom=8),
        )

        return card

    # ── Sniffing logic ──

    def packet_callback(pkt):
        nonlocal captured_count
        captured_count += 1
        flow_tracker.add_packet(pkt)

    def start_sniffing(e):
        nonlocal sniffer_thread, is_sniffing, captured_count

        if is_sniffing:
            return

        iface = iface_dropdown.value
        if not iface:
            add_log("ERROR: Select a network interface!", ACCENT_RED)
            return

        is_sniffing = True
        captured_count = 0
        flow_tracker.clear()
        status_text.value = "Status: Capturing..."
        status_text.color = ACCENT_GREEN
        start_btn.disabled = True
        stop_btn.disabled = False
        page.update()

        add_log(f"Started capturing on interface: {iface}", ACCENT_BLUE)

        def sniff_worker():
            try:
                sniff(
                    iface=iface,
                    prn=packet_callback,
                    store=False,
                    stop_filter=lambda _: not is_sniffing,
                )
            except PermissionError:
                add_log("ERROR: Run as Administrator/root for packet capture!", ACCENT_RED)
            except Exception as ex:
                add_log(f"ERROR: {str(ex)}", ACCENT_RED)

        sniffer_thread = threading.Thread(target=sniff_worker, daemon=True)
        sniffer_thread.start()

        # Update counters periodically
        def counter_updater():
            while is_sniffing:
                packets_counter.value = f"Packets: {captured_count}"
                flows_counter.value = f"Flows: {len(flow_tracker.flows)}"
                try:
                    page.update()
                except Exception:
                    break
                time.sleep(0.5)

        threading.Thread(target=counter_updater, daemon=True).start()

    def stop_sniffing(e):
        nonlocal is_sniffing
        is_sniffing = False
        status_text.value = "Status: Stopped"
        status_text.color = ACCENT_YELLOW
        start_btn.disabled = False
        stop_btn.disabled = True
        page.update()

        add_log(f"Stopped. Captured {captured_count} packets in {len(flow_tracker.flows)} flows.", ACCENT_YELLOW)

    def analyze_flows(e):
        """Extract features from captured flows and send to server."""
        server_url = server_url_field.value.rstrip("/")

        try:
            min_pkts = int(min_packets_field.value)
        except ValueError:
            min_pkts = 10

        results_column.controls.clear()
        analyzed = 0
        errors = 0

        add_log(f"Analyzing {len(flow_tracker.flows)} flows (min {min_pkts} packets)...", ACCENT_BLUE)

        for flow_key, flow_data in flow_tracker.flows.items():
            if len(flow_data["timestamps"]) < min_pkts:
                continue

            features = flow_tracker.extract_features(flow_key)
            if features is None:
                continue

            # Build readable flow key string
            (side_a, side_b, proto) = flow_key[0], flow_key[1], flow_key[2]
            proto_name = {6: "TCP", 17: "UDP"}.get(proto, str(proto))
            flow_key_str = f"{side_a[0]}:{side_a[1]} ↔ {side_b[0]}:{side_b[1]} ({proto_name})"

            try:
                response = requests.post(
                    f"{server_url}/analyze",
                    json={"features": features},
                    timeout=10,
                )
                data = response.json()

                if data.get("status") == "success":
                    card = build_result_card(flow_key_str, features, data)
                    results_column.controls.insert(0, card)
                    analyzed += 1
                else:
                    add_log(f"Server error for flow {flow_key_str}", ACCENT_RED)
                    errors += 1

            except requests.exceptions.ConnectionError:
                add_log(f"Cannot connect to server: {server_url}", ACCENT_RED)
                errors += 1
                break
            except Exception as ex:
                add_log(f"Error analyzing flow: {str(ex)}", ACCENT_RED)
                errors += 1

        add_log(f"Analysis complete: {analyzed} flows analyzed, {errors} errors.", ACCENT_GREEN if errors == 0 else ACCENT_YELLOW)
        page.update()

    # ── Buttons ──
    start_btn = ft.ElevatedButton(
        "Start Capture",
        icon=ft.icons.PLAY_ARROW_ROUNDED,
        on_click=start_sniffing,
        bgcolor=ACCENT_GREEN,
        color=BG_PRIMARY,
        style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8)),
        height=40,
    )

    stop_btn = ft.ElevatedButton(
        "Stop Capture",
        icon=ft.icons.STOP_ROUNDED,
        on_click=stop_sniffing,
        bgcolor=ACCENT_RED,
        color=BG_PRIMARY,
        disabled=True,
        style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8)),
        height=40,
    )

    analyze_btn = ft.ElevatedButton(
        "Analyze Flows",
        icon=ft.icons.ANALYTICS_ROUNDED,
        on_click=analyze_flows,
        bgcolor=ACCENT_BLUE,
        color=BG_PRIMARY,
        style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8)),
        height=40,
    )

    clear_btn = ft.TextButton(
        "Clear Results",
        icon=ft.icons.DELETE_OUTLINE_ROUNDED,
        on_click=lambda _: (results_column.controls.clear(), page.update()),
        style=ft.ButtonStyle(color=TEXT_SECONDARY),
    )

    # ── Layout ──

    # Header
    header = ft.Container(
        content=ft.Row(
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            controls=[
                ft.Row(
                    spacing=12,
                    controls=[
                        ft.Icon(ft.icons.SHIELD_ROUNDED, color=ACCENT_BLUE, size=28),
                        ft.Column(
                            spacing=0,
                            controls=[
                                ft.Text("Trojan Detection IDS", color=TEXT_PRIMARY, size=20, weight=ft.FontWeight.BOLD),
                                ft.Text("Network Traffic Analyzer", color=TEXT_SECONDARY, size=12),
                            ]
                        ),
                    ]
                ),
                ft.Row(spacing=8, controls=[status_text, packets_counter, flows_counter]),
            ]
        ),
        bgcolor=BG_CARD,
        border=ft.border.all(1, BORDER_COLOR),
        border_radius=10,
        padding=16,
    )

    # Controls panel
    controls_panel = ft.Container(
        content=ft.Column(
            spacing=12,
            controls=[
                ft.Text("Configuration", color=TEXT_PRIMARY, size=14, weight=ft.FontWeight.W_600),
                ft.Row(
                    spacing=12,
                    alignment=ft.MainAxisAlignment.START,
                    controls=[server_url_field, iface_dropdown, min_packets_field],
                ),
                ft.Row(
                    spacing=12,
                    controls=[start_btn, stop_btn, analyze_btn, clear_btn],
                ),
            ]
        ),
        bgcolor=BG_CARD,
        border=ft.border.all(1, BORDER_COLOR),
        border_radius=10,
        padding=16,
    )

    # Log panel
    log_panel = ft.Container(
        content=ft.Column(
            spacing=8,
            controls=[
                ft.Text("Event Log", color=TEXT_PRIMARY, size=14, weight=ft.FontWeight.W_600),
                ft.Container(
                    content=log_column,
                    bgcolor=BG_INPUT,
                    border=ft.border.all(1, BORDER_COLOR),
                    border_radius=8,
                    padding=10,
                ),
            ]
        ),
        bgcolor=BG_CARD,
        border=ft.border.all(1, BORDER_COLOR),
        border_radius=10,
        padding=16,
    )

    # Results panel
    results_panel = ft.Container(
        content=ft.Column(
            spacing=8,
            expand=True,
            controls=[
                ft.Text("Analysis Results", color=TEXT_PRIMARY, size=14, weight=ft.FontWeight.W_600),
                ft.Container(
                    content=results_column,
                    expand=True,
                ),
            ]
        ),
        bgcolor=BG_CARD,
        border=ft.border.all(1, BORDER_COLOR),
        border_radius=10,
        padding=16,
        expand=True,
    )

    # Final layout
    page.add(
        ft.Column(
            expand=True,
            spacing=12,
            controls=[
                header,
                controls_panel,
                log_panel,
                results_panel,
            ],
            scroll=ft.ScrollMode.AUTO,
        )
    )

    add_log("Application started. Select interface and press Start Capture.", ACCENT_BLUE)


if __name__ == "__main__":
    ft.app(target=main)