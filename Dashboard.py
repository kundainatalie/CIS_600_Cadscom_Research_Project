import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import pandas as pd

# LOAD DATA
cust = pd.read_csv("Enhanced_Retail_Analysis.csv", index_col=0)
ts   = pd.read_csv("monthly_timeseries.csv")

label_map = {
    "Loyal":       "High-Value Loyalists",
    "Mixed":       "Occasional Browsers",
    "Deal Seeker": "Seasonal Deal Seekers"
}
cust["Archetype"] = cust["Cluster_Label"].map(label_map)
ts["Archetype"]   = ts["Cluster_Label"].map(label_map)

# COLORS
C = {
    "High-Value Loyalists":   "#03045E",
    "Occasional Browsers":    "#0077B6",
    "Seasonal Deal Seekers":  "#00B4D8",
    "header": "#206e9e",
    "page":   "#FAF9F6",
    "card":   "#FFFFFF"
}

ARCH_COLORS  = [C["High-Value Loyalists"], C["Occasional Browsers"], C["Seasonal Deal Seekers"]]
cluster_order = ["High-Value Loyalists", "Occasional Browsers", "Seasonal Deal Seekers"]

app = dash.Dash(__name__)

# LAYOUT
app.layout = html.Div(className="page", children=[

    html.Div(className="header", children=[
        html.H2("Retail Behavioral Analytics Dashboard")
    ]),

    html.Div(className="container", children=[

        # KPI ROW
        html.Div(className="kpi-row", children=[
            html.Div(className="kpi-box", children=[
                html.H6("Total Customers"),
                html.H3(f"{len(cust):,}")
            ]),
            html.Div(className="kpi-box", children=[
                html.H6("Top Feature"),
                html.H3("Seasonality")
            ]),
            html.Div(className="kpi-box", children=[
                html.H6("Avg. Spend"),
                html.H3(f"£{cust['Monetary'].mean():,.0f}")
            ])
        ]),

        # FILTER
        html.Div(className="card", children=[
            html.H5("Filter by Loyalty (Tenure)"),
            dcc.Slider(
                id='loyalty-slider',
                min=1, max=12, step=1, value=1,
                marks={i: f'{i} Mo' for i in range(1, 13)}
            )
        ]),

        # TOGGLE — Raw vs Rolling
        html.Div(className="card", children=[
            html.H5("Revenue View"),
            dcc.RadioItems(
                id='ts-toggle',
                options=[
                    {'label': 'Raw Monthly Revenue',  'value': 'LineRevenue'},
                    {'label': '3-Month Rolling Avg',  'value': 'RollingRevenue'}
                ],
                value='RollingRevenue',
                inline=True
            )
        ]),

        # ROW 1
        html.Div(className="grid", children=[
            html.Div(className="card", children=[
                html.H5("Archetype Distribution"),
                dcc.Graph(id='pie-chart')
            ]),
            html.Div(className="card", children=[
                html.H5("Monthly Revenue Trend"),
                dcc.Graph(id='line-chart')
            ])
        ]),

        # ROW 2
        html.Div(className="card", children=[
            html.H5("Behavioral Profile Comparison"),
            dcc.Graph(id='profile-bar-chart')
        ])
    ])
])

# CALLBACK
@app.callback(
    [Output('pie-chart',         'figure'),
     Output('line-chart',        'figure'),
     Output('profile-bar-chart', 'figure')],
    [Input('loyalty-slider', 'value'),
     Input('pie-chart',      'clickData'),
     Input('ts-toggle',      'value')]
)
def update_dashboard(min_months, clickData, ts_mode):

    # FILTER customers by tenure
    dff = cust[cust['PurchaseSpread'] >= min_months].copy()

    if clickData:
        selected = clickData['points'][0]['label']
        dff = dff[dff['Archetype'] == selected]

    # ── PIE ──────────────────────────────────────────────────────────────
    summary = dff['Archetype'].value_counts().reset_index()

    fig_pie = go.Figure(go.Pie(
        labels=summary.iloc[:, 0],
        values=summary.iloc[:, 1],
        hole=.5,
        marker_colors=ARCH_COLORS,
        hovertemplate="<b>%{label}</b><br>%{value} customers<br>%{percent}<extra></extra>"
    ))

    # ── LINE — uses monthly_timeseries.csv (LineRevenue or RollingRevenue)
    fig_line = go.Figure()

    for arch, color in zip(cluster_order, ARCH_COLORS):
        df_arch = ts[ts['Archetype'] == arch].copy()
        df_arch = df_arch.sort_values("Month")

        if not df_arch.empty:
            fig_line.add_trace(go.Scatter(
                x=df_arch['Month'],
                y=df_arch[ts_mode],
                name=arch,
                mode='lines+markers',
                line=dict(color=color, width=3),
                hovertemplate="<b>%{x}</b><br>£%{y:,.0f}<extra></extra>"
            ))

    title_map = {
        "LineRevenue":    "Raw Monthly Revenue by Segment",
        "RollingRevenue": "3-Month Rolling Average Revenue by Segment"
    }
    fig_line.update_layout(
        title=title_map[ts_mode],
        xaxis_title="Month",
        yaxis_title="Revenue (£)",
        hovermode="x unified"
    )

    # ── BAR ──────────────────────────────────────────────────────────────
    features = ["Recency", "Frequency", "Monetary", "ReturnRate", "SeasonalConcentration"]

    fig_profile = go.Figure()

    for arch, color in zip(cluster_order, ARCH_COLORS):
        means = dff[dff["Archetype"] == arch][features].mean()
        norm  = [means[f] / (cust[f].max() + 1e-9) for f in features]

        fig_profile.add_trace(go.Bar(
            name=arch,
            x=features,
            y=norm,
            marker_color=color,
            hovertemplate="<b>%{x}</b><br>Score: %{y:.2f}<extra></extra>"
        ))

    fig_profile.update_layout(
        barmode='group',
        yaxis_title="Normalized Score"
    )

    return fig_pie, fig_line, fig_profile


if __name__ == "__main__":
    app.run(debug=True)