import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sympy.utilities.lambdify import lambdify

# 1. Definição das variáveis simbólicas
x, y, m, n = sp.symbols('x y m n')
a, b, t, E, nu, p0, P = sp.symbols('a b t E nu p0 P')

# Definindo a rigidez à flexão da placa (D)
D_expr = (E * t ** 3) / (12 * (1 - nu ** 2))


# 2. Definições comuns e funções de plotagem

def derive_and_plot(w_final_expr, params, example_name):
    """
    Takes a final symbolic deflection expression and a set of numerical
    parameters to derive all other quantities and generate plots.
    """
    print(f"\n--- Analysis for {example_name} ---")

    # --- Derive Symbolic Expressions ---
    print("Deriving symbolic expressions for moments and forces...")
    # Substitute D expression with a symbol for cleaner derivatives
    D_sym = sp.Symbol('D')

    # Momentos (mx, my, mxy)
    mx_expr = -D_sym * (sp.diff(w_final_expr, x, 2) + nu * sp.diff(w_final_expr, y, 2))
    my_expr = -D_sym * (sp.diff(w_final_expr, y, 2) + nu * sp.diff(w_final_expr, x, 2))
    mxy_expr = -D_sym * (1 - nu) * sp.diff(w_final_expr, x, y)

    # Cisalhamentos (Qx, Qy)
    Qx_expr = sp.diff(mx_expr, x) + sp.diff(mxy_expr, y)
    Qy_expr = sp.diff(mxy_expr, x) + sp.diff(my_expr, y)

    # Esforços cortantes efetivos (Vx, Vy)
    Vx_expr = Qx_expr + sp.diff(mxy_expr, y)
    Vy_expr = Qy_expr + sp.diff(mxy_expr, x)

    # Reações pontuais (R)
    R_expr = 2 * mxy_expr

    # --- Numerical Evaluation ---
    print("Generating plots...")

    # Calculate D value and add to params
    local_params = params.copy()
    D_val = D_expr.subs(local_params).evalf()
    local_params[D_sym] = D_val

    # Lambdify all expressions for fast numerical evaluation
    w_func = lambdify((x, y, a, b, D_sym, nu, p0, P), w_final_expr, 'numpy')
    mx_func = lambdify((x, y, a, b, D_sym, nu, p0, P), mx_expr, 'numpy')
    my_func = lambdify((x, y, a, b, D_sym, nu, p0, P), my_expr, 'numpy')
    mxy_func = lambdify((x, y, a, b, D_sym, nu, p0, P), mxy_expr, 'numpy')
    Qx_func = lambdify((x, y, a, b, D_sym, nu, p0, P), Qx_expr, 'numpy')
    Qy_func = lambdify((x, y, a, b, D_sym, nu, p0, P), Qy_expr, 'numpy')
    Vx_func = lambdify((x, y, a, b, D_sym, nu, p0, P), Vx_expr, 'numpy')
    Vy_func = lambdify((x, y, a, b, D_sym, nu, p0, P), Vy_expr, 'numpy')
    R_func = lambdify((x, y, a, b, D_sym, nu, p0, P), R_expr, 'numpy')

    # Create a meshgrid for plotting
    a_val, b_val = local_params[a], local_params[b]
    x_vals = np.linspace(0, a_val, 100)
    y_vals = np.linspace(0, b_val, 100)
    X, Y = np.meshgrid(x_vals, y_vals)

    # Calculate numerical values
    # Add dummy values for unused load type to prevent errors
    if p0 not in local_params: local_params[p0] = 0
    if P not in local_params: local_params[P] = 0

    W = w_func(X, Y, local_params[a], local_params[b], local_params[D_sym], local_params[nu], local_params[p0],
               local_params[P])
    Mx = mx_func(X, Y, local_params[a], local_params[b], local_params[D_sym], local_params[nu], local_params[p0],
                 local_params[P])
    My = my_func(X, Y, local_params[a], local_params[b], local_params[D_sym], local_params[nu], local_params[p0],
                 local_params[P])
    Mxy = mxy_func(X, Y, local_params[a], local_params[b], local_params[D_sym], local_params[nu], local_params[p0],
                   local_params[P])
    Qx = Qx_func(X, Y, local_params[a], local_params[b], local_params[D_sym], local_params[nu], local_params[p0],
                 local_params[P])
    Qy = Qy_func(X, Y, local_params[a], local_params[b], local_params[D_sym], local_params[nu], local_params[p0],
                 local_params[P])

    # --- Plotting ---
    plot_3d_surface(X, Y, W * 1000, 'Deflection w(x,y)', 'Deflection (mm)')
    plot_3d_surface(X, Y, Mx / 1000, 'Moment $m_x$', 'Moment (kN-m/m)')
    plot_3d_surface(X, Y, My / 1000, 'Moment $m_y$', 'Moment (kN-m/m)')
    plot_3d_surface(X, Y, Mxy / 1000, 'Twisting Moment $m_{xy}$', 'Moment (kN-m/m)')
    plot_3d_surface(X, Y, Qx / 1000, 'Shear Force $Q_x$', 'Force/Length (kN/m)')
    plot_3d_surface(X, Y, Qy / 1000, 'Shear Force $Q_y$', 'Force/Length (kN/m)')

    plot_edge_reactions(x_vals, y_vals, Vx_func, Vy_func, local_params)
    demonstrate_corner_forces(R_func, local_params)


def plot_3d_surface(X, Y, Z, title, zlabel):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', rstride=2, cstride=2)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    plt.show()


def plot_edge_reactions(x_vals, y_vals, Vx_func, Vy_func, params):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    fig.suptitle('Reactions Along Plate Edges', fontsize=16)

    a_val, b_val = params[a], params[b]

    Vx_at_xa = Vx_func(a_val, y_vals, **params)
    axs[0].plot(y_vals, Vx_at_xa / 1000, label=f'$V_x$ at x={a_val}')
    axs[0].set_title('Reaction $V_x$ Along Edge x=a')
    axs[0].set_xlabel('y (m)')
    axs[0].set_ylabel('Reaction (kN/m)')
    axs[0].legend()
    axs[0].grid(True)

    Vy_at_yb = Vy_func(x_vals, b_val, **params)
    axs[1].plot(x_vals, Vy_at_yb / 1000, label=f'$V_y$ at y={b_val}')
    axs[1].set_title('Reaction $V_y$ Along Edge y=b')
    axs[1].set_xlabel('x (m)')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def demonstrate_corner_forces(R_func, params):
    print("\n--- Demonstration of Corner Forces ---")
    R_at_00 = R_func(0, 0, **params)
    R_at_ab = R_func(params[a], params[b], **params)
    print("Concentrated forces R are required at the corners to maintain equilibrium.")
    print(f"The formula is R = 2 * m_xy.")
    print(f"Calculated corner force at (x=0, y=0): R = {R_at_00:.2f} N")
    print(f"Calculated corner force at (x={params[a]}, y={params[b]}): R = {R_at_ab:.2f} N")


# --- 3. EXAMPLE-SPECIFIC SOLVERS ---

def solve_example_5_1():
    """ Plate under Uniform Pressure """
    params = {a: 3.0, b: 3.0, t: 0.01, E: 200e9, nu: 0.3, p0: 10e3, P: 0}
    D_val = D_expr.subs(params)

    # Fourier coefficient for uniform load p0
    Q_mn_expr = (16 * p0) / (m * n * sp.pi ** 2)
    A_mn_expr = Q_mn_expr / (sp.pi ** 4 * D_val * (m ** 2 / a ** 2 + n ** 2 / b ** 2) ** 2)
    w_mn_expr = A_mn_expr * sp.sin(m * sp.pi * x / a) * sp.sin(n * sp.pi * y / b)

    print("--- Part (a): Convergence Analysis for Uniform Load ---")
    w_max_prev = 0
    w_final_expr = 0

    for i in range(1, 10):
        max_terms = 2 * i - 1
        w_series = 0
        for m_val in range(1, max_terms + 1, 2):
            for n_val in range(1, max_terms + 1, 2):
                w_series += w_mn_expr.subs({m: m_val, n: n_val})

        w_func_conv = lambdify((x, y), w_series.subs(params), 'numpy')
        w_max_current = w_func_conv(params[a] / 2, params[b] / 2)

        if w_max_prev != 0:
            diff = abs((w_max_current - w_max_prev) / w_max_prev)
            print(f"Terms up to m,n={max_terms}: w_max = {w_max_current * 1000:.4f} mm, Change = {diff:.2%}")
            if diff < 0.05:
                w_final_expr = w_series
                print(f"\nSolution converged with terms up to (m,n) = {max_terms}.")
                break
        else:
            print(f"Terms up to m,n={max_terms}: w_max = {w_max_current * 1000:.4f} mm")
        w_max_prev = w_max_current

    if w_final_expr == 0: w_final_expr = w_series  # Ensure it's assigned if loop finishes

    derive_and_plot(w_final_expr, params, "Example 5.1: Uniform Load")


def solve_example_5_2():
    """ Plate under Sinusoidal Load """
    params = {a: 2.0, b: 3.0, t: 0.01, E: 200e9, nu: 0.3, p0: 10e3, P: 0}
    D_val = D_expr.subs(params)

    print(
        "\nFor a sinusoidal load p(x,y) = p0*sin(pi*x/a)*sin(pi*y/b), the solution is exact and consists of a single term (m=1, n=1).")
    print("No convergence analysis is needed.")

    # The solution is exact with a single term
    A_11_expr = p0 / (sp.pi ** 4 * D_val * (1 / a ** 2 + 1 / b ** 2) ** 2)
    w_final_expr = A_11_expr * sp.sin(sp.pi * x / a) * sp.sin(sp.pi * y / b)

    derive_and_plot(w_final_expr, params, "Example 5.2: Sinusoidal Load")


def solve_example_5_3():
    """ Plate under Concentrated Load at Center """
    # Parameters from Example 5.3
    params = {a: 0.600, b: 0.600, t: 0.010, E: 70e9, nu: 0.3, P: 5e3, p0: 0}
    D_val = D_expr.subs(params)

    # Q_mn for concentrated load P at center
    Q_mn_expr = (4 * P / (a * b)) * sp.sin(m * sp.pi / 2) * sp.sin(n * sp.pi / 2)
    A_mn_expr = Q_mn_expr / (sp.pi ** 4 * D_val * (m ** 2 / a ** 2 + n ** 2 / b ** 2) ** 2)
    w_mn_expr = A_mn_expr * sp.sin(m * sp.pi * x / a) * sp.sin(n * sp.pi * y / b)

    print("--- Part (a): Convergence Analysis for Concentrated Load ---")
    w_max_prev = 0
    w_final_expr = 0

    # Use more iterations as concentrated loads converge slower
    for i in range(1, 15):
        max_terms = 2 * i - 1
        w_series = 0
        for m_val in range(1, max_terms + 1, 2):
            for n_val in range(1, max_terms + 1, 2):
                w_series += w_mn_expr.subs({m: m_val, n: n_val})

        w_func_conv = lambdify((x, y), w_series.subs(params), 'numpy')
        w_max_current = w_func_conv(params[a] / 2, params[b] / 2)

        if w_max_prev != 0:
            diff = abs((w_max_current - w_max_prev) / w_max_prev)
            print(f"Terms up to m,n={max_terms}: w_max = {w_max_current * 1000:.4f} mm, Change = {diff:.2%}")
            if diff < 0.05:
                w_final_expr = w_series
                print(f"\nSolution converged with terms up to (m,n) = {max_terms}.")
                break
        else:
            print(f"Terms up to m,n={max_terms}: w_max = {w_max_current * 1000:.4f} mm")
        w_max_prev = w_max_current

    if w_final_expr == 0: w_final_expr = w_series

    derive_and_plot(w_final_expr, params, "Example 5.3: Concentrated Load")

# --- 4. MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    while True:
        print("\n" + "=" * 50)
        print("Select the example to run:")
        print("  1: Example 5.1 (Uniform Pressure)")
        print("  2: Example 5.2 (Sinusoidal Load)")
        print("  3: Example 5.3 (Concentrated Load at Center)")
        print("  q: Quit")
        choice = input("Enter your choice (1, 2, 3, or q): ")

        if choice == '1':
            solve_example_5_1()
        elif choice == '2':
            solve_example_5_2()
        elif choice == '3':
            solve_example_5_3()
        elif choice.lower() == 'q':
            print("Exiting.")
            break
        else:
            print("Invalid choice, please try again.")
        print("=" * 50)