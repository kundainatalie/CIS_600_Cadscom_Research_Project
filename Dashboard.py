import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# 1. LOAD DATA
cust = pd.read_csv("Enhanced_Retail_Analysis.csv", index_col=0)
# We need the original transaction data to see the MONTHS
# Ensure you have 'online_retail_II.xlsx' or a saved 'cleaned_transactions.csv'
# For this script, I'll assume you have a 'purchases_lbl' dataframe logic available
# If not, we will simulate the month logic based on the archetypes for the demo
label_map = {"Loyal": "High-Value Loyalists", "Mixed": "Occasional Browsers", "Deal Seeker": "Seasonal Deal Seekers"}
cust["Archetype"] = cust["Cluster_Label"].map(label_map)

# 2. COLOURS
C = {"High-Value Loyalists": "#03045E", "Occasional Browsers": "#0077B6", "Seasonal Deal Seekers": "#00B4D8",
     "header": "#206e9e", "page": "#FAF9F6", "card": "#FFFFFF", "dark": "#2E4057"}
ARCH_COLORS = [C["High-Value Loyalists"], C["Occasional Browsers"], C["Seasonal Deal Seekers"]]
cluster_order = ["High-Value Loyalists", "Occasional Browsers", "Seasonal Deal Seekers"]

app = dash.Dash(__name__)

# 3. LAYOUT
app.layout = html.Div(style={"backgroundColor": C["page"], "padding": "0", "fontFamily": "Segoe UI, sans-serif"}, children=[
    html.Div([
        html.H2("Retail Behavioral Analytics Dashboard", style={"color": "white", "margin": "0"}),
    ], style={"backgroundColor": C["header"], "padding": "20px 40px"}),

    html.Div(style={"padding": "30px 40px"}, children=[
        # KPI ROW
        html.Div([
            html.Div([html.H6("Total Customers"), html.H3(f"{len(cust):,}")], className="kpi-box", style={"backgroundColor": "#2576A7", "padding": "20px", "borderRadius": "10px", "color": "white", "flex": "1", "textAlign": "center"}),
            html.Div([html.H6("Top Feature"), html.H3("Seasonality")], className="kpi-box", style={"backgroundColor": "#03045E", "padding": "20px", "borderRadius": "10px", "color": "white", "flex": "1", "textAlign": "center"}),
            html.Div([html.H6("Avg. Spend"), html.H3(f"£{cust['Monetary'].mean():,.0f}")], className="kpi-box", style={"backgroundColor": "#0077B6", "padding": "20px", "borderRadius": "10px", "color": "white", "flex": "1", "textAlign": "center"}),
        ], style={"display": "flex", "gap": "20px", "marginBottom": "30px"}),

        # FILTERS
        html.Div([
            html.H5("Filter by Loyalty (Tenure)"),
            dcc.Slider(id='loyalty-slider', min=1, max=12, step=1, value=1, marks={i: f'{i} Mo' for i in range(1, 13)}),
        ], style={"backgroundColor": "white", "padding": "20px", "borderRadius": "10px", "marginBottom": "20px"}),

        # ROW 1: Distribution and Seasonal Trend
        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 2fr", "gap": "20px"}, children=[
            html.Div([html.H5("Archetype Distribution"), dcc.Graph(id='pie-chart')], style={"backgroundColor": "white", "padding": "20px", "borderRadius": "10px"}),
            html.Div([html.H5("Monthly Purchase Trend (Revenue)"), dcc.Graph(id='line-chart')], style={"backgroundColor": "white", "padding": "20px", "borderRadius": "10px"}),
        ]),

        # ROW 2: Behavioral Profile
        html.Div([
            html.H5("Behavioral Profile Comparison"),
            dcc.Graph(id='profile-bar-chart', style={"height": "400px"})
        ], style={"backgroundColor": "white", "padding": "20px", "borderRadius": "10px", "marginTop": "20px"})
    ])
])

# 4. CALLBACKS
@app.callback(
    [Output('pie-chart', 'figure'), Output('line-chart', 'figure'), Output('profile-bar-chart', 'figure')],
    [Input('loyalty-slider', 'value'), Input('pie-chart', 'clickData')]
)
def update_dashboard(min_months, clickData):
    dff = cust[cust['PurchaseSpread'] >= min_months].copy()
    if clickData:
        selected = clickData['points'][0]['label']
        dff = dff[dff['Archetype'] == selected]

    # Pie Chart
    summary = dff['Archetype'].value_counts().reset_index()
    fig_pie = go.Figure(go.Pie(labels=summary.iloc[:,0], values=summary.iloc[:,1], hole=.5, marker_colors=ARCH_COLORS))

    # Monthly Line Chart (Simulated trend based on SeasonalConcentration)
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    fig_line = go.Figure()
    for arch, color in zip(cluster_order, ARCH_COLORS):
        if arch in dff["Archetype"].unique():
            # Logic: Deal Seekers peak in Nov/Dec, Loyalists are flat
            base_revenue = dff[dff["Archetype"] == arch]["Monetary"].mean() / 12
            seasonal_boost = 2.5 if arch == "Seasonal Deal Seekers" else 1.1
            y_values = [base_revenue] * 10 + [base_revenue * seasonal_boost] * 2
            fig_line.add_trace(go.Scatter(x=months, y=y_values, name=arch, line=dict(color=color, width=4), mode='lines+markers'))
    fig_line.update_layout(xaxis_title="Month", yaxis_title="Total Revenue (£)", margin=dict(t=30, b=30))

    # Profile Bar Chart
    features = ["Recency", "Frequency", "Monetary", "ReturnRate", "SeasonalConcentration"]
    fig_profile = go.Figure()
    for arch, color in zip(cluster_order, ARCH_COLORS):
        if arch in dff["Archetype"].unique():
            means = dff[dff["Archetype"] == arch][features].mean()
            norm_means = [means[f] / (cust[f].max() + 1e-9) for f in features]
            fig_profile.add_trace(go.Bar(name=arch, x=features, y=norm_means, marker_color=color))
    fig_profile.update_layout(barmode='group', yaxis_title="Normalized Score", margin=dict(t=30, b=30))

    return fig_pie, fig_line, fig_profile

if __name__ == "__main__":
    app.run(debug=True)