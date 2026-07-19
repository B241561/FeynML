"""Shared chart color palette for server-side Plotly visualizations.

Place centralized color definitions here so multiple modules can import
the same palette and avoid drift.
"""

FEYNML_CHART_COLORS = {
    'primary': '#FFB020',    # amber — main data series
    'secondary': '#3D5A73',  # steel — comparison/reference series
    'critical': '#E23E3E',   # failure/error states
    'high': '#E8C547',       # warning/high severity (distinct yellow-amber)
    'stable': '#4FB286',     # pass/healthy states
}
