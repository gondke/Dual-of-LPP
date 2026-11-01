import streamlit as st
import numpy as np
from scipy.optimize import linprog
import plotly.graph_objects as go
from scipy.spatial import ConvexHull

st.set_page_config(
    page_title="LPP Duality & Visualization (2D/3D)",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Linear Programming Duality & Geometric Visualization")
st.markdown("""
This app demonstrates **Primal-Dual Duality** in Linear Programming.
It converts a Primal LPP into its Dual formulation and provides **2D or 3D interactive plots** of both feasible regions to illustrate why:
1. **Weak Duality holds**: $Z_{\text{primal}} \le W_{\text{dual}}$ for any feasible point.
2. **Strong Duality holds**: $Z^*_{\text{primal}} = W^*_{\text{dual}}$ at optimal points (if an optimal solution exists).
""")

# --- SIDEBAR INPUTS ---
st.sidebar.header("1. Primal LPP Configuration")

opt_type = st.sidebar.radio("Objective Type", ["Maximize", "Minimize"])
num_vars = st.sidebar.number_input("Number of Decision Variables", min_value=2, max_value=3, value=2, step=1)
num_constraints = st.sidebar.number_input("Number of Constraints", min_value=2, max_value=3, value=3, step=1)

# Objective Function Coefficients
st.sidebar.subheader("Objective Coefficients ($c$)")
c = []
cols_c = st.sidebar.columns(num_vars)
default_c = [3.0, 5.0, 4.0]
for i in range(num_vars):
    val = cols_c[i].number_input(f"c_{i+1}", value=float(default_c[i if i < 3 else 0]))
    c.append(val)
c = np.array(c)

# Constraints
st.sidebar.subheader("Constraints ($A \cdot x \le b$ or $\ge b$)")
A = []
b = []
constraint_types = []

default_A = [[1.0, 2.0, 1.0], [1.0, 1.0, 1.0], [2.0, 1.0, 1.0]]
default_b = [6.0, 4.0, 6.0]

for j in range(num_constraints):
    st.sidebar.markdown(f"**Constraint {j+1}**")
    cols_a = st.sidebar.columns(num_vars + 2)
    row = []
    for i in range(num_vars):
        a_val = cols_a[i].number_input(
            f"A_{{{j+1},{i+1}}}", 
            value=float(default_A[j if j < 3 else 0][i if i < 3 else 0])
        )
        row.append(a_val)
    op = cols_a[num_vars].selectbox(f"Op {j+1}", ["<=", ">="], index=0)
    rhs = cols_a[num_vars+1].number_input(f"b_{j+1}", value=float(default_b[j if j < 3 else 0]))
    
    A.append(row)
    b.append(rhs)
    constraint_types.append(op)

A = np.array(A)
b = np.array(b)

# Canonical form setup (Maximize c^T x s.t. A_ub x <= b_ub, x >= 0)
A_ub = A.copy()
b_ub = b.copy()

for j in range(num_constraints):
    if constraint_types[j] == ">=":
        A_ub[j] = -A_ub[j]
        b_ub[j] = -b_ub[j]

if opt_type == "Minimize":
    c_primal_canonical = -c
else:
    c_primal_canonical = c.copy()

# --- DUAL FORMULATION ---
# Primal: Max c^T x s.t. A_ub x <= b_ub, x >= 0
# Dual:   Min b_ub^T y s.t. A_ub^T y >= c_primal_canonical, y >= 0
# Represent Dual as standard <= form: -A_ub^T y <= -c_primal_canonical
b_dual = c_primal_canonical
A_dual_ub = -A_ub.T
c_dual = b_ub

# --- SOLVE PRIMAL & DUAL ---
res_primal = linprog(-c_primal_canonical, A_ub=A_ub, b_ub=b_ub, bounds=(0, None), method='highs')
res_dual = linprog(c_dual, A_ub=A_dual_ub, b_ub=-b_dual, bounds=(0, None), method='highs')

# --- DISPLAY FORMULATIONS ---
col1, col2 = st.columns(2)

with col1:
    st.header("Primal Problem")
    st.latex(rf"\text{{{opt_type} }} Z = " + " + ".join([f"{c[i]}x_{i+1}" for i in range(num_vars)]))
    st.write("**Subject to:**")
    for j in range(num_constraints):
        expr = " + ".join([f"{A[j, i]}x_{i+1}" for i in range(num_vars)])
        st.latex(rf"{expr} {constraint_types[j]} {b[j]}")
    st.latex(rf"x_1, \dots, x_{num_vars} \ge 0")

    if res_primal.success:
        st.success("**Optimal Primal Solution:**")
        for i in range(num_vars):
            st.write(f"$x_{i+1}^* = {res_primal.x[i]:.4f}$")
        opt_val_p = -res_primal.fun if opt_type == "Maximize" else res_primal.fun
        st.write(f"**Optimal Value ($Z^*$):** {opt_val_p:.4f}")
    else:
        st.error("Primal Problem is Infeasible or Unbounded.")

with col2:
    st.header("Dual Problem")
    st.latex(rf"\text{{Minimize }} W = " + " + ".join([f"{b_ub[j]}y_{j+1}" for j in range(num_constraints)]))
    st.write("**Subject to:**")
    for i in range(num_vars):
        expr = " + ".join([f"{A_ub[j, i]}y_{j+1}" for j in range(num_constraints)])
        st.latex(rf"{expr} \ge {c_primal_canonical[i]}")
    st.latex(rf"y_1, \dots, y_{num_constraints} \ge 0")

    if res_dual.success:
        st.success("**Optimal Dual Solution:**")
        for j in range(num_constraints):
            st.write(f"$y_{j+1}^* = {res_dual.x[j]:.4f}$")
        st.write(f"**Optimal Value ($W^*$):** {res_dual.fun:.4f}")
    else:
        st.error("Dual Problem is Infeasible or Unbounded.")

st.divider()

# --- THEORETICAL INSIGHTS ---
st.subheader("Why Duality Exists & Why $Z^* = W^*$")
st.markdown("""
- **Economic Valuation:** The Primal optimizes product output $x$ given resource limits $b$. The Dual optimizes shadow prices $y$ assigned to resources $b$ to cover costs.
- **Weak Duality:** For any feasible $x$ and $y$, $Z(x) = c^T x \le y^T A x \le y^T b = W(y)$. The dual objective acts as an upper bound to the primal objective.
- **Strong Duality:** At optimum, the feasible bounds align perfectly, eliminating slack and making $Z^* = W^*$.
""")

# --- ROBUST VISUALIZATION FUNCTIONS ---

def plot_2d_region(A_ineq, b_ineq, title, var_names, opt_point):
    fig = go.Figure()

    # Dynamic upper range based on optimal point or constraints
    max_range = 10.0
    if opt_point is not None and len(opt_point) >= 2:
        max_range = max(10.0, np.max(opt_point) * 2.5)

    x1_vals = np.linspace(0, max_range, 200)
    x2_vals = np.linspace(0, max_range, 200)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)

    feasible_mask = (X1 >= 0) & (X2 >= 0)
    for row_a, val_b in zip(A_ineq, b_ineq):
        feasible_mask &= (row_a[0] * X1 + row_a[1] * X2 <= val_b)

        if abs(row_a[1]) > 1e-5:
            y_line = (val_b - row_a[0] * x1_vals) / row_a[1]
            valid = (y_line >= 0) & (y_line <= max_range)
            fig.add_trace(go.Scatter(
                x=x1_vals[valid], y=y_line[valid], mode='lines',
                name=f"{row_a[0]:.1f}{var_names[0]} + {row_a[1]:.1f}{var_names[1]} <= {val_b:.1f}"
            ))
        else:
            if abs(row_a[0]) > 1e-5:
                x_val = val_b / row_a[0]
                if 0 <= x_val <= max_range:
                    fig.add_trace(go.Scatter(
                        x=[x_val, x_val], y=[0, max_range], mode='lines',
                        name=f"{row_a[0]:.1f}{var_names[0]} <= {val_b:.1f}"
                    ))

    # Draw shaded feasible region contour
    fig.add_trace(go.Contour(
        z=feasible_mask.astype(int),
        x=x1_vals,
        y=x2_vals,
        showscale=False,
        colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(0, 150, 255, 0.4)']],
        name="Feasible Region",
        hoverinfo='skip'
    ))

    if opt_point is not None and len(opt_point) >= 2:
        fig.add_trace(go.Scatter(
            x=[opt_point[0]], y=[opt_point[1]],
            mode='markers+text',
            marker=dict(color='red', size=14, symbol='star'),
            text=["Optimum"],
            textposition="top right",
            name="Optimum"
        ))

    fig.update_layout(
        title=title,
        xaxis_title=var_names[0],
        yaxis_title=var_names[1],
        xaxis=dict(range=[0, max_range]),
        yaxis=dict(range=[0, max_range]),
        height=500
    )
    return fig


def plot_3d_region(A_ineq, b_ineq, title, var_names, opt_point):
    fig = go.Figure()

    max_range = 8.0
    if opt_point is not None and len(opt_point) == 3:
        max_range = max(8.0, np.max(opt_point) * 2.0)

    # Grid sampling to extract valid interior vertices for 3D Mesh
    grid = np.linspace(0, max_range, 25)
    V1, V2, V3 = np.meshgrid(grid, grid, grid)

    feasible_mask = (V1 >= 0) & (V2 >= 0) & (V3 >= 0)
    for row_a, val_b in zip(A_ineq, b_ineq):
        feasible_mask &= (row_a[0] * V1 + row_a[1] * V2 + row_a[2] * V3 <= val_b)

    pts = np.vstack((V1[feasible_mask], V2[feasible_mask], V3[feasible_mask])).T

    if len(pts) > 4:
        try:
            hull = ConvexHull(pts)
            fig.add_trace(go.Mesh3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                i=hull.simplices[:, 0], j=hull.simplices[:, 1], k=hull.simplices[:, 2],
                color='cyan', opacity=0.4, name="Feasible Region"
            ))
        except Exception:
            fig.add_trace(go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode='markers', marker=dict(size=3, color='blue', opacity=0.3),
                name="Feasible Points"
            ))
    else:
        st.warning(f"Feasible region for {title} is bounded or outside default rendering window.")

    if opt_point is not None and len(opt_point) == 3:
        fig.add_trace(go.Scatter3d(
            x=[opt_point[0]], y=[opt_point[1]], z=[opt_point[2]],
            mode='markers+text',
            marker=dict(color='red', size=10, symbol='diamond'),
            text=["Optimum"],
            textposition="top center",
            name="Optimum"
        ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=var_names[0],
            yaxis_title=var_names[1],
            zaxis_title=var_names[2],
            xaxis=dict(range=[0, max_range]),
            yaxis=dict(range=[0, max_range]),
            zaxis=dict(range=[0, max_range]),
        ),
        height=500
    )
    return fig


# --- RENDER GRAPHICAL PLOTS ---
st.header("Geometric Visualizations (2D / 3D)")

vcol1, vcol2 = st.columns(2)

with vcol1:
    st.subheader("Primal Feasible Region")
    if num_vars == 2:
        fig_p = plot_2d_region(A_ub, b_ub, "Primal Region (2D)", ["x1", "x2"], res_primal.x if res_primal.success else None)
        st.plotly_chart(fig_p, use_container_width=True)
    elif num_vars == 3:
        fig_p = plot_3d_region(A_ub, b_ub, "Primal Region (3D)", ["x1", "x2", "x3"], res_primal.x if res_primal.success else None)
        st.plotly_chart(fig_p, use_container_width=True)

with vcol2:
    st.subheader("Dual Feasible Region")
    num_dual_vars = num_constraints  # Dual dimension = number of primal constraints
    if num_dual_vars == 2:
        fig_d = plot_2d_region(A_dual_ub, -b_dual, "Dual Region (2D)", ["y1", "y2"], res_dual.x if res_dual.success else None)
        st.plotly_chart(fig_d, use_container_width=True)
    elif num_dual_vars == 3:
        fig_d = plot_3d_region(A_dual_ub, -b_dual, "Dual Region (3D)", ["y1", "y2", "y3"], res_dual.x if res_dual.success else None)
        st.plotly_chart(fig_d, use_container_width=True)
