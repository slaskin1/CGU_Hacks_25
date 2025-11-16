from flask import Flask, render_template, request, url_for
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot

# Use current folder for templates
app = Flask(__name__, template_folder='.')

# ---------- LOAD DATA (ALL WAVES) ----------

WAVE_FILES = [
    ("Wave 1", "Wave1Unbiased.csv"),
    ("Wave 2", "Wave2Unbiased.csv"),
    ("Wave 3", "Wave3Unbiased.csv"),
    ("Wave 4", "Wave4Unbiased.csv"),
    ("Wave 5", "Wave5Unbiased.csv"),
    ("Wave 6", "Wave6Unbiased.csv"),
    ("Wave 7", "Wave7Unbiased.csv"),
]

frames = []
for wave_label, path in WAVE_FILES:
    tmp = pd.read_csv(path)
    tmp["Wave"] = wave_label
    frames.append(tmp)

df = pd.concat(frames, ignore_index=True)

if "Entry Date and Time" in df.columns:
    df["Entry Date and Time"] = pd.to_datetime(
        df["Entry Date and Time"], errors="coerce"
    )

# ---------- HELPERS FOR WAVES ----------

def sort_waves(waves):
    def key_fn(w):
        s = str(w)
        try:
            return int(s.strip().split()[-1])
        except Exception:
            return 999
    return sorted(waves, key=key_fn)

wave_labels = sort_waves(df["Wave"].dropna().unique().tolist())

# ---------- METRICS (1–5) ----------

METRIC_COLS = {
    "Attendance Scores": "Attendance",
    "Route Effeciency": "Route Efficiency",
    "Driving Behavior": "Driving Behavior",
    "Safety Check": "Safety",
    "Peer Feedback Averages": "Peer Feedback",
    "Customer Feedback": "Customer Feedback",
    "Compliance": "Compliance",
    "Vehicle Maintenance": "Vehicle Maint.",
    "Load Securing": "Load Securing",
    "Technical Proficiency": "Technical",
}

for col in METRIC_COLS.keys():
    df[col] = pd.to_numeric(df[col], errors="coerce")

employee_names = sorted(df["Name"].dropna().unique().tolist())
supervisor_names = sorted(df["Supervisor"].dropna().unique().tolist())

# ---------- CORE HELPERS ----------

def get_record_for_employee_wave(df, name, wave_label):
    sub = df[(df["Name"] == name) & (df["Wave"] == wave_label)].copy()
    if sub.empty:
        return None
    if "Entry Date and Time" in sub.columns:
        sub = sub.sort_values("Entry Date and Time")
    return sub.iloc[-1]

# BRAND COLORS
RED = "#b40123"
RED_LIGHT = "rgba(180,1,35,0.2)"

# palette for supervisor view (6 distinct colors)
SUP_COLORS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#17becf",  # teal
]

# ---------- FIGURE FUNCTIONS ----------

# ===== WORKER VIEW (RED THEME) =====

def employee_radar_figure(row, wave_label):
    categories = list(METRIC_COLS.values())
    values = [row[col] for col in METRIC_COLS.keys()]

    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values_closed,
            theta=categories_closed,
            fill="toself",
            fillcolor=RED_LIGHT,
            line_color=RED,
            marker_color=RED,
            name=row.get("Name", "Employee"),
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[1, 5], gridcolor="#ddd"),
            angularaxis=dict(gridcolor="#ddd"),
        ),
        font=dict(color=RED),
        title=f"Skill Profile – {row.get('Name', '')} – {wave_label}",
        showlegend=False,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def employee_timeseries_figure(df, name):
    metric_cols = list(METRIC_COLS.keys())
    y_values = []

    for wave in wave_labels:
        sub = df[(df["Name"] == name) & (df["Wave"] == wave)]
        if sub.empty:
            y_values.append(None)
        else:
            avg_per_row = sub[metric_cols].mean(axis=1)
            y_values.append(avg_per_row.mean())

    x_values = [str(w) for w in wave_labels]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=y_values,
            mode="lines+markers",
            name=f"{name} avg score",
            connectgaps=False,
            marker=dict(size=8, color=RED),
            line=dict(color=RED, width=3),
        )
    )

    fig.update_layout(
        title=f"Average Skill Score Over Time – {name}",
        xaxis_title="Wave",
        yaxis_title="Average of All Metrics (1–5)",
        yaxis=dict(range=[0.5, 5]),
        hovermode="x unified",
        font=dict(color=RED),
    )
    return fig


# ===== SUPERVISOR VIEW (MULTI-COLOR) =====

def supervisor_radar_figure(df, supervisor, wave_label):
    sub = df[(df["Supervisor"] == supervisor) & (df["Wave"] == wave_label)].copy()
    if sub.empty:
        return go.Figure()

    if "Entry Date and Time" in sub.columns:
        sub = sub.sort_values("Entry Date and Time")
        latest = sub.groupby("Name").tail(1)
    else:
        latest = sub.groupby("Name").tail(1)

    categories = list(METRIC_COLS.values())
    categories_closed = categories + [categories[0]]

    fig = go.Figure()

    for i, (_, row) in enumerate(latest.iterrows()):
        color = SUP_COLORS[i % len(SUP_COLORS)]
        values = [row[col] for col in METRIC_COLS.keys()]
        values_closed = values + [values[0]]

        fig.add_trace(
            go.Scatterpolar(
                r=values_closed,
                theta=categories_closed,
                fill="toself",
                fillcolor=color,      # different fill per employee
                line_color=color,
                marker_color=color,
                opacity=0.3,
                name=row.get("Name", "Employee"),
            )
        )

    fig.update_layout(
        title=f"Employees under {supervisor} – {wave_label}",
        polar=dict(radialaxis=dict(visible=True, range=[1, 5])),
        showlegend=True,
        margin=dict(l=40, r=40, t=60, b=40),
        font=dict(color=RED),   # keep titles/labels on-brand
    )
    return fig


def supervisor_timeseries_figure(df, supervisor):
    metric_cols = list(METRIC_COLS.keys())
    sup_df = df[df["Supervisor"] == supervisor].copy()
    if sup_df.empty:
        return go.Figure()

    fig = go.Figure()
    x_values = [str(w) for w in wave_labels]

    for i, emp_name in enumerate(sorted(sup_df["Name"].dropna().unique().tolist())):
        color = SUP_COLORS[i % len(SUP_COLORS)]
        y_values = []
        for wave in wave_labels:
            sub = sup_df[(sup_df["Name"] == emp_name) & (sup_df["Wave"] == wave)]
            if sub.empty:
                y_values.append(None)
            else:
                avg_per_row = sub[metric_cols].mean(axis=1)
                y_values.append(avg_per_row.mean())

        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="lines+markers",
                name=emp_name,
                connectgaps=False,
                line=dict(color=color, width=2),
                marker=dict(color=color),
            )
        )

    fig.update_layout(
        title=f"Average Skill Score Over Time – Employees under {supervisor}",
        xaxis_title="Wave",
        yaxis_title="Average of All Metrics (1–5)",
        yaxis=dict(range=[0.5, 5]),
        legend_title="Employee",
        font=dict(color=RED),   # titles/axes still red
    )
    return fig


# ---------- ROUTES ----------

@app.route("/")
@app.route("/worker")
def worker_dashboard():
    default_employee = employee_names[0] if employee_names else ""
    default_wave = wave_labels[0] if wave_labels else ""

    selected_employee = request.args.get("employee", default_employee)
    selected_wave = request.args.get("wave", default_wave)

    row = get_record_for_employee_wave(df, selected_employee, selected_wave)
    if row is not None:
        emp_radar_fig = employee_radar_figure(row, selected_wave)
        emp_radar_div = plot(emp_radar_fig, output_type="div", include_plotlyjs=False)
    else:
        emp_radar_div = "<p>No data for selected employee/wave.</p>"

    emp_ts_fig = employee_timeseries_figure(df, selected_employee)
    emp_ts_div = plot(emp_ts_fig, output_type="div", include_plotlyjs=False)

    nav_supervisor = supervisor_names[0] if supervisor_names else ""

    return render_template(
        "dashtemplate.html",
        view="worker",
        employee_names=employee_names,
        supervisor_names=supervisor_names,
        wave_labels=wave_labels,
        selected_employee=selected_employee,
        selected_supervisor=nav_supervisor,
        selected_wave=selected_wave,
        emp_radar_div=emp_radar_div,
        emp_ts_div=emp_ts_div,
        sup_radar_div="",
        sup_ts_div="",
    )


@app.route("/supervisor")
def supervisor_dashboard():
    default_supervisor = supervisor_names[0] if supervisor_names else ""
    default_wave = wave_labels[0] if wave_labels else ""

    selected_supervisor = request.args.get("supervisor", default_supervisor)
    selected_wave = request.args.get("wave", default_wave)

    sup_radar_fig = supervisor_radar_figure(df, selected_supervisor, selected_wave)
    sup_radar_div = plot(sup_radar_fig, output_type="div", include_plotlyjs=False)

    sup_ts_fig = supervisor_timeseries_figure(df, selected_supervisor)
    sup_ts_div = plot(sup_ts_fig, output_type="div", include_plotlyjs=False)

    nav_employee = employee_names[0] if employee_names else ""

    return render_template(
        "dashtemplate.html",
        view="supervisor",
        employee_names=employee_names,
        supervisor_names=supervisor_names,
        wave_labels=wave_labels,
        selected_employee=nav_employee,
        selected_supervisor=selected_supervisor,
        selected_wave=selected_wave,
        emp_radar_div="",
        emp_ts_div="",
        sup_radar_div=sup_radar_div,
        sup_ts_div=sup_ts_div,
    )


if __name__ == "__main__":
    app.run(debug=True)
