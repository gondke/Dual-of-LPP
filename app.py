import streamlit as st
import numpy as np
from scipy.optimize import linprog
import plotly.graph_objects as go

st.set_page_config(
    page_title="LPP Duality & Visualization",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Linear Programming Duality & Geometric Visualization")
st.markdown("""
This app demonstrates **Primal-Dual Duality** in Linear Programming. 
It converts a Primal LPP into its Dual formulation and provides 2D/3D geometric plots of both feasible regions to illustrate why:
1. **Weak Duality holds**: $Z_{\text{primal}} \le W_{\text{dual}}$ for any feasible solution.
2. **Strong Duality holds**: $Z^*_{\text{primal}} = W^*_{\text{dual}}$ at optimal points (if an optimal solution exists).
""")

# --- SIDEBAR INPUTS ---
st.sidebar.header("1. Primal LPP Configuration")

opt_type = st.sidebar.radio("Objective Type", ["Maximize", "Minimize"])
num_vars = st.sidebar.number_input("Number of Decision Variables", min_value=2, max_value=3, value=2, step=1)
num_constraints = st.sidebar.number_input("Number of Constraints", min_value=1, max_value=5, value=2, step=1)

# Input Objective Function Coefficients
st.sidebar.subheader("Objective Function Coefficients ($c$)")
c = []
cols_c = st.sidebar.columns(num_vars)
for i in range(num_vars):
    val = cols_c[i].number_input(f"c_{i+1}", value=float([3.0, 5.0, 4.0][i if i < 3 else 0]))
    c.append(val)

c = np.array(c)

# Input Constraints
st.sidebar.subheader("Constraints ($A \cdot x \le b$ or $\ge b$)")
A = []
b = []
constraint_types = []

for j in range(num_constraints):
    st.sidebar.markdown(f"**Constraint {j+1}**")
    cols_a = st.sidebar.columns(num_vars + 2)
    row = []
    for i in range(num_vars):
        a_val = cols_a[i].number_input(f"A_{{{j+1},{i+1}}}", value=float([[1.0, 0.0], [0.0, 2.0], [3.0, 2.0]][j if j < 3 else 0][i if i < 2 else 0]))
        row.append(a_val)
    op = cols_a[num_vars].selectbox(f"Op {j+1}", ["<=", ">="], index=0)
    rhs = cols_a[num_vars+1].number_input(f"b_{j+1}", value=float([4.0, 12.0, 18.0][j if j < 3 else 0]))
    
    A.append(row)
    b.append(rhs)
    constraint_types.append(op)

A = np.array(A)
b = np.array(b)

# Standardize Primal to Canonical form (Maximize c^T x s.t. A x <= b, x >= 0)
# If user selected Minimize or >=, adjust signs accordingly for duality derivation
A_ub = A.copy()
b_ub = b.copy()

for j in range(num_constraints):
    if constraint_types[j] == ">=":
        A_ub[j] = -A_ub[j]
        b_ub[j] = -b_ub[j]

if opt_type == "Minimize":
    # Min c^T x <=> Max (-c)^T x
    c_primal_canonical = -c
else:
    c_primal_canonical = c.copy()

# --- DUAL FORMULATION ---
# Primal: Max c^T x s.t. A_ub x <= b_ub, x >= 0
# Dual:   Min b_ub^T y s.t. A_ub^T y >= c_primal_canonical, y >= 0
b_dual = c_primal_canonical
A_dual_ub = -A_ub.T
c_dual = b_ub

# --- SOLVING PRIMAL & DUAL ---
# Solve Primal
res_primal = linprog(-c_primal_canonical, A_ub=A_ub, b_ub=b_ub, bounds=(0, None), method='highs')
# Solve Dual
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
        st.success(f"**Optimal Primal Solution:**")
        for i in range(num_vars):
            st.write(f"$x_{i+1}^* = {res_primal.x[i]:.4f}$")
        st.write(f"**Optimal Value ($Z^*$):** { -res_primal.fun if opt_type=='Maximize' else res_primal.fun:.4f}")
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
        st.success(f"**Optimal Dual Solution:**")
        for j in range(num_constraints):
            st.write(f"$y_{j+1}^* = {res_dual.x[j]:.4f}$")
        st.write(f"**Optimal Value ($W^*$):** {res_dual.fun:.4f}")
    else:
        st.error("Dual Problem is Infeasible or Unbounded.")

st.divider()

# --- THEORETICAL INSIGHTS ---
st.subheader("Why Duality Exists & Why $Z^* = W^*$")
st.markdown("""
- **Resource Valuation (Economic Interpretation):** While the Primal maximizes profit from production choices $x$, the Dual minimizes the shadow prices $y$ assigned to resources $b$.
- **Weak Duality:** For any feasible primal $x$ and dual $y$, $c^T x \le (A^T y)^T x = y^T A x \le y^T b$. Thus, dual objective value $W(y)$ acts as an upper bound for primal objective value $Z(x)$.
- **Strong Duality:** At the optimal vertex, no improving direction exists in the primal feasible space, forcing the gap between $c^T x^*$ and $b^T y^*$ to shrink to zero ($Z^* = W^*$).
""")

# --- VISUALIZATIONS ---
st.header("Geometric Visualizations")

def plot_2d_region(A_ineq, b_ineq, title, var_names, opt_point):
    fig = go.Figure()
    x1_vals = np.linspace(0, 10, 200)
    x2_vals = np.linspace(0, 10, 200)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)
    
    feasible_mask = (X1 >= 0) & (X2 >= 0)
    for row_a, val_b in zip(A_ineq, b_ineq):
        feasible_mask &= (row_a[0]*X1 + row_a[1]*X2 <= val_b)
        
        # Plot constraint line
        if abs(row_a[1]) > 1e-5:
            y_line = (val_b - row_a[0]*x1_vals) / row_a[1]
            fig.add_trace(go.Scatter(x=x1_vals, y=y_line, mode='lines', name=f"{row_a[0]}x1 + {row_a[1]}x2 <= {val_b}"))
        else:
            x_val = val_b / row_a[0]
            fig.add_trace(go.Scatter(x=[x_val, x_val], y=[0, 10], mode='lines', name=f"{row_a[0]}x1 <= {val_b}"))

    # Draw feasible region contour
    fig.add_trace(go.Contour(
        z=feasible_mask.astype(int),
        x=x1_vals,
        y=x2_vals,
        showscale=False,
        colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(0, 150, 255, 0.3)']],
        name="Feasible Region"
    ))

    if opt_point is not None and len(opt_point) >= 2:
        fig.add_trace(go.Scatter(
            x=[opt_point[0]], y=[opt_point[1]],
            mode='markers+text',
            marker=dict(color='red', size=12),
            text=["Optimal"],
            textposition="top right",
            name="Optimum"
        ))

    fig.update_layout(
        title=title,
        xaxis_title=var_names[0],
        yaxis_title=var_names[1],
        xaxis=dict(range=[0, 8]),
        yaxis=dict(range=[0, 8]),
        width=500, height=450
    )
    return fig

vcol1, vcol2 = st.columns(2)

with vcol1:
    if num_vars == 2:
        fig_p = plot_2d_region(A_ub, b_ub, "Primal Feasible Region (2D)", ["x1", "x2"], res_primal.x if res_primal.success else None)
        st.plotly_chart(fig_p, use_container_width=True)
    else:
        st.info("Primal visualization is restricted to 2D when number of variables > 2.")

with vcol2:
    if num_constraints == 2:
        fig_d = plot_2d_region(A_dual_ub, -b_dual, "Dual Feasible Region (2D)", ["y1", "y2"], res_dual.x if res_dual.success else None)
        st.plotly_chart(fig_d, use_container_width=True)
    else:
        st.info("Dual visualization requires exactly 2 constraints to display in 2D.")
