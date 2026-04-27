import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
cust = pd.read_csv("Enhanced_Retail_Analysis.csv", index_col=0)
ts   = pd.read_csv("monthly_timeseries.csv")

label_map = {
    "Loyal":       "High-Value Loyalists",
    "Mixed":       "Occasional Browsers",
    "Deal Seeker": "Seasonal Deal Seekers"
}
cust["Archetype"] = cust["Cluster_Label"].map(label_map)
ts["Archetype"]   = ts["Cluster_Label"].map(label_map)

# ── COLORS ────────────────────────────────────────────────────────────────────
SEG_COLORS = {
    "High-Value Loyalists":  "#03045E",
    "Occasional Browsers":   "#0077B6",
    "Seasonal Deal Seekers": "#00B4D8"
}
cluster_order = ["High-Value Loyalists", "Occasional Browsers", "Seasonal Deal Seekers"]
ARCH_COLORS   = [SEG_COLORS[a] for a in cluster_order]

# ── PRE-COMPUTE SUMMARY STATS ─────────────────────────────────────────────────
nov_dec = [11, 12]
seg_map = {"High-Value Loyalists": "Loyal", "Occasional Browsers": "Mixed",
           "Seasonal Deal Seekers": "Deal Seeker"}

summary_stats = []
for arch in cluster_order:
    lbl = seg_map[arch]
    s   = ts[ts["Cluster_Label"] == lbl].set_index("Month")["LineRevenue"]
    s.index = pd.to_datetime(s.index)
    mean_r  = s.mean()
    std_r   = s.std()
    cv      = std_r / mean_r if mean_r > 0 else 0
    nd_avg  = s[s.index.month.isin(nov_dec)].mean()
    base    = s[~s.index.month.isin(nov_dec)].mean()
    ratio   = nd_avg / base if base > 0 else float("nan")
    summary_stats.append({
        "Archetype": arch,
        "Mean £":    f"£{mean_r:,.0f}",
        "CV":        f"{cv:.3f}",
        "Peak Ratio":f"{ratio:.2f}x"
    })

# ── KPI VALUES ────────────────────────────────────────────────────────────────
total_customers  = len(cust)
deal_seeker_pct  = (cust["Cluster_Label"] == "Deal Seeker").mean()
avg_spend        = cust["Monetary"].mean()
top_feature      = "Seasonal Conc."

# ── APP ───────────────────────────────────────────────────────────────────────
app = dash.Dash(__name__)

app.layout = html.Div(children=[

    # HEADER
    html.Div(className="header", children=[
        html.H2("Retail Behavioral Analytics"),
        #html.Span("UCI Online Retail II · 2010–2011", className="header-badge")
    ]),

    # BODY
    html.Div(className="body-layout", children=[

        # ── SIDEBAR ──────────────────────────────────────────────────────────
        html.Div(className="sidebar", children=[

            html.Div(className="sidebar-section", children=[
                html.H6("Filter by Tenure"),
                dcc.Slider(
                    id='loyalty-slider',
                    min=1, max=12, step=1, value=1,
                    marks={1: '1', 3: '3', 6: '6', 9: '9', 12: '12'},
                    tooltip={"placement": "bottom", "always_visible": False}
                )
            ]),

            html.Div(className="sidebar-section", children=[
                html.H6("Revenue View"),
                dcc.RadioItems(
                    id='ts-toggle',
                    className="radio-inline",
                    options=[
                        {'label': 'Monthly',   'value': 'LineRevenue'},
                        {'label': '3-Mo Rolling',  'value': 'RollingRevenue'}
                    ],
                    value='RollingRevenue',
                    labelStyle={'display': 'block', 'marginBottom': '8px',
                                'fontSize': '0.83rem'}
                )
            ]),

            html.Div(className="sidebar-section", children=[
                html.H6("Segment Filter"),
                dcc.Checklist(
                    id='seg-filter',
                    options=[{'label': f' {a}', 'value': a} for a in cluster_order],
                    value=cluster_order,
                    labelStyle={'display': 'block', 'marginBottom': '8px',
                                'fontSize': '0.83rem'}
                )
            ]),

            # Legend dots
            html.Div(className="sidebar-section", children=[
                html.H6("Segments"),
                html.Div([
                    html.Div([
                        html.Span(className="seg-dot",
                                  style={"backgroundColor": SEG_COLORS[a]}),
                        html.Span(a, style={"fontSize": "0.78rem"})
                    ], style={"marginBottom": "8px", "display": "flex",
                              "alignItems": "center"})
                    for a in cluster_order
                ])
            ])
        ]),

        # ── MAIN CONTENT ─────────────────────────────────────────────────────
        html.Div(className="main-content", children=[

            # KPI ROW
            html.Div(className="kpi-row", children=[
                html.Div(className="kpi-box", children=[
                    html.H6("Total Customers"),
                    html.H3(f"{total_customers:,}"),
                    html.P("analysed", className="kpi-sub")
                ]),
                html.Div(className="kpi-box", children=[
                    html.H6("Deal Seekers"),
                    html.H3(f"{deal_seeker_pct:.0%}"),
                    html.P("of customer base", className="kpi-sub")
                ]),
                html.Div(className="kpi-box", children=[
                    html.H6("Avg. Spend"),
                    html.H3(f"£{avg_spend:,.0f}"),
                    html.P("per customer", className="kpi-sub")
                ]),
                html.Div(className="kpi-box", children=[
                    html.H6("Top Discriminator"),
                    html.H3(top_feature),
                    html.P("RF importance 31.9%", className="kpi-sub")
                ])
            ]),

            # ROW 1 — Pie + Line
            html.Div(className="grid-2", children=[
                html.Div(className="card", children=[
                    html.H5("Archetype Distribution"),
                    dcc.Graph(id='pie-chart',
                              config={'displayModeBar': False},
                              style={"height": "300px"})
                ]),
                html.Div(className="card", children=[
                    html.H5("Monthly Revenue Trend"),
                    dcc.Graph(id='line-chart',
                              config={'displayModeBar': False},
                              style={"height": "300px"})
                ])
            ]),

            # ROW 2 — Bar + Stats table
            html.Div(className="grid-2", children=[
                html.Div(className="card", children=[
                    html.H5("Behavioral Profile Comparison"),
                    dcc.Graph(id='profile-bar-chart',
                              config={'displayModeBar': False},
                              style={"height": "300px"})
                ]),
                html.Div(className="card", children=[
                    html.H5("Time Series Summary Statistics"),
                    html.Table(className="stats-table", children=[
                        html.Thead(html.Tr([
                            html.Th("Segment"),
                            html.Th("Mean Rev."),
                            html.Th("CV"),
                            html.Th("Peak Ratio")
                        ])),
                        html.Tbody([
                            html.Tr([
                                html.Td([
                                    html.Span(className="seg-dot",
                                              style={"backgroundColor":
                                                     SEG_COLORS[row["Archetype"]]}),
                                    row["Archetype"].split()[0]  # first word only
                                ]),
                                html.Td(row["Mean £"]),
                                html.Td(row["CV"]),
                                html.Td(row["Peak Ratio"])
                            ])
                            for row in summary_stats
                        ])
                    ])
                ])
            ]),

        ])
    ]),

    # FOOTER
    html.Div(className="footer",
             children="K-Means · Random Forest · Time Series · by Kundai Chirimumimba · CIS 602  · Spring 2026 ")
])


# ── CALLBACK ──────────────────────────────────────────────────────────────────
@app.callback(
    [Output('pie-chart',         'figure'),
     Output('line-chart',        'figure'),
     Output('profile-bar-chart', 'figure')],
    [Input('loyalty-slider', 'value'),
     Input('pie-chart',      'clickData'),
     Input('ts-toggle',      'value'),
     Input('seg-filter',     'value')]
)
def update_dashboard(min_months, clickData, ts_mode, selected_segs):

    selected_segs = selected_segs or cluster_order

    # Filter customers
    dff = cust[
        (cust['PurchaseSpread'] >= min_months) &
        (cust['Archetype'].isin(selected_segs))
    ].copy()

    if clickData:
        clicked = clickData['points'][0]['label']
        if clicked in selected_segs:
            dff = dff[dff['Archetype'] == clicked]

    # ── PIE ──────────────────────────────────────────────────────────────────
    summary = dff['Archetype'].value_counts().reindex(cluster_order, fill_value=0).reset_index()
    summary.columns = ["Archetype", "Count"]

    fig_pie = go.Figure(go.Pie(
        labels=summary["Archetype"],
        values=summary["Count"],
        hole=0.55,
        marker=dict(colors=ARCH_COLORS, line=dict(color='white', width=2)),
        textinfo='percent',
        hovertemplate="<b>%{label}</b><br>%{value} customers (%{percent})<extra></extra>"
    ))
    fig_pie.update_layout(
        margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        font=dict(family='DM Sans')
    )

    # ── LINE ─────────────────────────────────────────────────────────────────
    fig_line = go.Figure()

    # Shade Nov-Dec
    fig_line.add_vrect(
        x0="2010-11", x1="2010-12",
        fillcolor="#f87171", opacity=0.08,
        layer="below", line_width=0,
        annotation_text="Nov–Dec", annotation_position="top left",
        annotation_font_size=10, annotation_font_color="#f87171"
    )
    fig_line.add_vrect(
        x0="2011-11", x1="2011-12",
        fillcolor="#f87171", opacity=0.08,
        layer="below", line_width=0
    )

    for arch, color in zip(cluster_order, ARCH_COLORS):
        if arch not in selected_segs:
            continue
        lbl     = seg_map[arch]
        df_arch = ts[ts["Cluster_Label"] == lbl].copy().sort_values("Month")
        col     = ts_mode

        # Faint raw line
        fig_line.add_trace(go.Scatter(
            x=df_arch["Month"], y=df_arch["LineRevenue"],
            mode='lines', name=f"{arch} (raw)",
            line=dict(color=color, width=1),
            opacity=0.25, showlegend=False,
            hoverinfo='skip'
        ))
        # Bold rolling / selected line
        fig_line.add_trace(go.Scatter(
            x=df_arch["Month"], y=df_arch[col],
            mode='lines+markers', name=arch,
            line=dict(color=color, width=2.5),
            marker=dict(size=5),
            hovertemplate=f"<b>{arch}</b><br>%{{x}}<br>£%{{y:,.0f}}<extra></extra>"
        ))

    fig_line.update_layout(
        margin=dict(t=10, b=30, l=10, r=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, tickangle=-30, tickfont=dict(size=10)),
        yaxis=dict(showgrid=True, gridcolor='#f1f5f9',
                   tickprefix="£", tickfont=dict(size=10)),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    font=dict(size=10)),
        font=dict(family='DM Sans')
    )

    # ── BAR ──────────────────────────────────────────────────────────────────
    features      = ["Recency", "Frequency", "Monetary", "ReturnRate", "SeasonalConcentration"]
    feature_labels = ["Recency", "Frequency", "Monetary", "Return Rate", "Seasonal Conc."]

    fig_profile = go.Figure()
    for arch, color in zip(cluster_order, ARCH_COLORS):
        if arch not in selected_segs:
            continue
        means = dff[dff["Archetype"] == arch][features].mean()
        norm  = [means[f] / (cust[f].max() + 1e-9) for f in features]
        fig_profile.add_trace(go.Bar(
            name=arch, x=feature_labels, y=norm,
            marker_color=color,
            marker_line=dict(color='white', width=1),
            hovertemplate="<b>%{x}</b><br>%{y:.3f}<extra></extra>"
        ))

    fig_profile.update_layout(
        barmode='group',
        margin=dict(t=10, b=30, l=10, r=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(showgrid=True, gridcolor='#f1f5f9',
                   title="Normalized Score", tickfont=dict(size=10)),
        xaxis=dict(tickfont=dict(size=10)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    font=dict(size=10)),
        font=dict(family='DM Sans')
    )

    return fig_pie, fig_line, fig_profile


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)