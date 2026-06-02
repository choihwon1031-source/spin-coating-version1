import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Spin Coating Thin-Film Simulator",
    layout="wide"
)

# =====================================================
# Basic functions
# =====================================================

def rpm_to_omega(rpm):
    return 2.0 * np.pi * rpm / 60.0


def eta_meyerhofer(t, eta0, B):
    return eta0 * np.exp(B * t)


def uniformity_percent(h):
    h_avg = np.mean(h)
    if h_avg <= 0:
        return np.nan
    return (np.max(h) - np.min(h)) / (2.0 * h_avg) * 100.0


def predict_t_gel(B, gel_factor=20.0):
    if B <= 0:
        return None
    return np.log(gel_factor) / B


def ebp_analytical(h0, rho, rpm, eta0, t):
    omega = rpm_to_omega(rpm)
    return h0 / np.sqrt(
        1.0 + (4.0 * rho * omega**2 * h0**2 / (3.0 * eta0)) * t
    )


# =====================================================
# EBP numerical model
# =====================================================

def simulate_ebp(h0, rho, rpm, eta0, t_end, dt):
    omega = rpm_to_omega(rpm)
    t = np.arange(0.0, t_end + dt, dt)
    h = np.zeros_like(t)
    h[0] = h0

    for n in range(len(t) - 1):
        dhdt = -(2.0 * rho * omega**2 / (3.0 * eta0)) * h[n]**3
        h[n + 1] = max(h[n] + dt * dhdt, 0.0)

    return t, h


# =====================================================
# Meyerhofer-type model
# =====================================================

def simulate_meyerhofer(h0, rho, rpm, eta0, B, E, h_dry, t_end, dt):
    omega = rpm_to_omega(rpm)
    t = np.arange(0.0, t_end + dt, dt)
    h = np.zeros_like(t)
    eta = np.zeros_like(t)

    h[0] = h0

    for n in range(len(t) - 1):
        eta[n] = eta_meyerhofer(t[n], eta0, B)

        dhdt = -(2.0 * rho * omega**2 / (3.0 * eta[n])) * h[n]**3 - E
        h[n + 1] = max(h[n] + dt * dhdt, h_dry)

    eta[-1] = eta_meyerhofer(t[-1], eta0, B)

    return t, h, eta


# =====================================================
# Semi-empirical radial uniformity model
# =====================================================

def simulate_radial_model(
    h0,
    rho,
    rpm,
    eta0,
    B,
    E,
    h_dry,
    R,
    t_end,
    dt,
    Nr,
    edge_bead_strength,
    edge_bead_width_ratio,
    leveling_coeff
):
    r = np.linspace(0.0, R, Nr)
    t, h_mean, eta_t = simulate_meyerhofer(
        h0=h0,
        rho=rho,
        rpm=rpm,
        eta0=eta0,
        B=B,
        E=E,
        h_dry=h_dry,
        t_end=t_end,
        dt=dt
    )

    edge_width = max(edge_bead_width_ratio * R, 1e-12)
    edge_shape = np.exp(-((R - r) / edge_width) ** 2)
    bead_shape = edge_shape - np.mean(edge_shape)

    rpm_ref = 3000.0
    eta_ref = 0.05

    leveling_rate = (
        leveling_coeff
        * 0.030
        * (rpm / rpm_ref) ** 2
        * (eta_ref / eta0)
    )

    profiles = []
    profile_time = []
    rows = []

    save_stride = max(1, int(1.0 / dt))

    for n, time in enumerate(t):
        eta_now = eta_t[n]

        amplitude = edge_bead_strength * np.exp(-leveling_rate * time)

        h_profile = h_mean[n] * (1.0 + amplitude * bead_shape)
        h_profile = np.maximum(h_profile, h_dry)

        if n % save_stride == 0 or n == len(t) - 1:
            profiles.append(h_profile.copy())
            profile_time.append(time)

        rows.append({
            "time_s": time,
            "center_h_um": h_profile[0] * 1e6,
            "middle_h_um": h_profile[Nr // 2] * 1e6,
            "edge_h_um": h_profile[-1] * 1e6,
            "avg_h_um": np.mean(h_profile) * 1e6,
            "uniformity_percent": uniformity_percent(h_profile),
            "eta_Pa_s": eta_now,
            "leveling_rate_1_s": leveling_rate
        })

    data = pd.DataFrame(rows)

    return r, np.array(profiles), np.array(profile_time), data, leveling_rate


# =====================================================
# Challenge search
# =====================================================

def challenge_search(
    spec,
    rpm_min,
    rpm_max,
    eta_min,
    eta_max,
    h0,
    rho,
    B,
    E,
    h_dry,
    R,
    t_end,
    dt,
    Nr,
    edge_bead_strength,
    edge_bead_width_ratio,
    leveling_coeff
):
    rpm_values = np.linspace(rpm_min, rpm_max, 16)
    eta_values = np.linspace(eta_min, eta_max, 16)

    rows = []

    for rpm_i in rpm_values:
        for eta_i in eta_values:
            r, profiles, profile_time, radial_data, leveling_rate = simulate_radial_model(
                h0=h0,
                rho=rho,
                rpm=rpm_i,
                eta0=eta_i,
                B=B,
                E=E,
                h_dry=h_dry,
                R=R,
                t_end=t_end,
                dt=dt,
                Nr=Nr,
                edge_bead_strength=edge_bead_strength,
                edge_bead_width_ratio=edge_bead_width_ratio,
                leveling_coeff=leveling_coeff
            )

            final_u = radial_data["uniformity_percent"].iloc[-1]

            rows.append({
                "RPM": rpm_i,
                "omega_rad_s": rpm_to_omega(rpm_i),
                "eta0_Pa_s": eta_i,
                "leveling_rate_1_s": leveling_rate,
                "final_avg_thickness_um": radial_data["avg_h_um"].iloc[-1],
                "final_uniformity_percent": final_u,
                "meets_spec": final_u <= spec
            })

    df = pd.DataFrame(rows)
    success = df[df["meets_spec"] == True].copy()

    if len(success) > 0:
        success = success.sort_values(
            by=["final_uniformity_percent", "RPM", "eta0_Pa_s"],
            ascending=[True, True, True]
        )

    return df, success


# =====================================================
# Sidebar
# =====================================================

st.sidebar.title("Input Parameters")

rpm = st.sidebar.slider("Spin Speed RPM", 0, 8000, 3000, 100)
h0_um = st.sidebar.slider("Initial Thickness h₀ [μm]", 1.0, 100.0, 10.0, 1.0)
eta0 = st.sidebar.slider("Initial Viscosity η₀ [Pa·s]", 0.01, 0.50, 0.05, 0.01)
rho = st.sidebar.number_input("Density ρ [kg/m³]", value=1000.0, step=50.0)

E_um_s = st.sidebar.slider("Evaporation Rate E [μm/s]", 0.00, 0.20, 0.03, 0.01)
B = st.sidebar.slider("Viscosity Growth Rate B [1/s]", 0.00, 0.10, 0.03, 0.005)

gel_factor = 20.0
eta_gel = gel_factor * eta0

h_dry_um = st.sidebar.slider("Dry Film Thickness Limit h_dry [μm]", 0.10, 5.00, 0.50, 0.10)
R_cm = st.sidebar.slider("Wafer Radius R [cm]", 1.0, 10.0, 5.0, 0.5)

edge_bead_strength = st.sidebar.slider("Edge Bead Strength", 0.00, 0.30, 0.08, 0.01)
edge_bead_width_ratio = st.sidebar.slider("Edge Bead Width Ratio", 0.02, 0.30, 0.12, 0.01)
leveling_coeff = st.sidebar.slider("Radial Leveling Coefficient", 0.10, 10.00, 2.00, 0.10)

t_end = st.sidebar.slider("Simulation Time [s]", 5.0, 120.0, 60.0, 5.0)
dt = st.sidebar.selectbox("Time Step Δt [s]", [0.02, 0.05, 0.10], index=1)
Nr = st.sidebar.slider("Radial Grid Number", 40, 160, 80, 10)

st.sidebar.markdown("---")
st.sidebar.title("Challenge Mode")

spec = st.sidebar.number_input("Uniformity Spec ± [%]", value=2.00, step=0.10)
rpm_min = st.sidebar.number_input("Search RPM min", value=1000.0, step=100.0)
rpm_max = st.sidebar.number_input("Search RPM max", value=6000.0, step=100.0)
eta_min = st.sidebar.number_input("Search η₀ min [Pa·s]", value=0.02, step=0.01)
eta_max = st.sidebar.number_input("Search η₀ max [Pa·s]", value=0.20, step=0.01)


# =====================================================
# Unit conversion
# =====================================================

h0 = h0_um * 1e-6
E = E_um_s * 1e-6
h_dry = h_dry_um * 1e-6
R = R_cm * 1e-2


# =====================================================
# Main calculations
# =====================================================

t_ebp, h_ebp = simulate_ebp(h0, rho, rpm, eta0, t_end, dt)
h_ebp_exact = ebp_analytical(h0, rho, rpm, eta0, t_ebp)

t_mey, h_mey, eta_mey = simulate_meyerhofer(
    h0=h0,
    rho=rho,
    rpm=rpm,
    eta0=eta0,
    B=B,
    E=E,
    h_dry=h_dry,
    t_end=t_end,
    dt=dt
)

r, profiles, profile_time, radial_data, leveling_rate = simulate_radial_model(
    h0=h0,
    rho=rho,
    rpm=rpm,
    eta0=eta0,
    B=B,
    E=E,
    h_dry=h_dry,
    R=R,
    t_end=t_end,
    dt=dt,
    Nr=Nr,
    edge_bead_strength=edge_bead_strength,
    edge_bead_width_ratio=edge_bead_width_ratio,
    leveling_coeff=leveling_coeff
)

final_profile = profiles[-1]
final_uniformity = radial_data["uniformity_percent"].iloc[-1]

t_gel = predict_t_gel(B, gel_factor)


# =====================================================
# Main UI
# =====================================================

st.title("Spin Coating Thin-Film Simulator")
st.caption(
    "EBP + Meyerhofer model, validation, radial visualization, gel-time prediction, and challenge-mode process design"
)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Final Thickness: EBP", f"{h_ebp[-1] * 1e6:.3f} μm")
col2.metric("Final Thickness: Meyerhofer", f"{h_mey[-1] * 1e6:.3f} μm")
col3.metric("Difference", f"{abs(h_mey[-1] - h_ebp[-1]) * 1e6:.3f} μm")
col4.metric("Final η(t)", f"{eta_mey[-1]:.3f} Pa·s")

col5, col6, col7, col8 = st.columns(4)

col5.metric("Radial Uniformity", f"±{final_uniformity:.3f} %")
col6.metric("Angular Velocity ω", f"{rpm_to_omega(rpm):.1f} rad/s")
col7.metric("η_gel = 20η₀", f"{eta_gel:.3f} Pa·s")

if t_gel is None:
    col8.metric("t_gel Prediction", "Not reached")
elif t_gel > t_end:
    col8.metric("t_gel Prediction", f">{t_end:.1f} s")
else:
    col8.metric("t_gel Prediction", f"{t_gel:.2f} s")


tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "EBP vs Meyerhofer",
    "Radial Animation",
    "Validation View",
    "Challenge Mode",
    "Design Exploration",
    "Process Insight",
    "Simulation Data"
])


# =====================================================
# Tab 1
# =====================================================

with tab1:
    st.subheader("Thickness Evolution")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_ebp, h_ebp * 1e6, color="red", linewidth=3, label="EBP Numerical")
    ax.plot(t_mey, h_mey * 1e6, color="cyan", linewidth=3, label="Meyerhofer")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Thickness [μm]")
    ax.set_title("EBP vs Meyerhofer Thickness Evolution")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    st.pyplot(fig)

    st.markdown("### Governing Equations")
    st.latex(r"\omega = \frac{2\pi RPM}{60}")
    st.latex(r"\frac{dh}{dt}=-\frac{2\rho\omega^2}{3\eta}h^3")
    st.latex(r"\eta(t)=\eta_0 e^{Bt}")
    st.latex(r"\frac{dh}{dt}=-\frac{2\rho\omega^2}{3\eta_0e^{Bt}}h^3-E")
    st.latex(r"\eta_{gel}=20\eta_0")
    st.latex(r"t_{gel}=\frac{\ln(20)}{B}")


# =====================================================
# Tab 2
# =====================================================

with tab2:
    st.subheader("Real-time Visualization of h(r,t)")

    idx = st.slider("Select animation time", 0, len(profile_time) - 1, len(profile_time) - 1)
    selected_profile = profiles[idx]

    st.write(f"Time = {profile_time[idx]:.2f} s")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(r * 100.0, selected_profile * 1e6, color="magenta", linewidth=3, label="h(r,t)")
    ax.set_xlabel("Radial Position r [cm]")
    ax.set_ylabel("Film Thickness h [μm]")
    ax.set_title("Radial Film Thickness Profile")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)

    st.subheader("Radial Uniformity vs Time")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        radial_data["time_s"],
        radial_data["uniformity_percent"],
        color="yellow",
        linewidth=3,
        label="Radial Uniformity"
    )
    ax.axhline(
        spec,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Spec ±{spec:.2f}%"
    )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Uniformity ± [%]")
    ax.set_title("Radial Uniformity Evolution")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)


# =====================================================
# Tab 3
# =====================================================

with tab3:
    st.subheader("Validation View")

    st.markdown("### Validation A: Numerical EBP vs Analytical EBP")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_ebp, h_ebp * 1e6, color="lime", linewidth=3, label="EBP Numerical")
    ax.plot(t_ebp, h_ebp_exact * 1e6, color="orange", linewidth=3, linestyle="--", label="EBP Analytical")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Thickness [μm]")
    ax.set_title("Analytical Solution Comparison")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)

    error = np.max(np.abs(h_ebp - h_ebp_exact)) * 1e6
    st.metric("Maximum Validation Error", f"{error:.6f} μm")

    st.markdown("### Validation B: Analytical Limit Checks")

    val_rows = []

    # ω -> 0 check: use RPM=0 and E=0
    t_v1, h_v1 = simulate_ebp(h0, rho, 0, eta0, t_end, dt)
    val_rows.append({
        "Limit": "ω → 0",
        "Input condition": "RPM = 0, E = 0",
        "Expected result": "dh/dt → 0",
        "Simulator result": f"Δh = {(h_v1[-1]-h_v1[0])*1e6:.6f} μm"
    })

    # η -> infinity check: use large viscosity
    eta_large = 1e9
    t_v2, h_v2 = simulate_ebp(h0, rho, rpm, eta_large, t_end, dt)
    val_rows.append({
        "Limit": "η → ∞",
        "Input condition": "η = 1e9 Pa·s",
        "Expected result": "dh/dt → 0",
        "Simulator result": f"Δh = {(h_v2[-1]-h_v2[0])*1e6:.6f} μm"
    })

    # E -> 0 and B -> 0 check: Meyerhofer -> EBP
    t_v3_e, h_v3_e = simulate_ebp(h0, rho, rpm, eta0, t_end, dt)
    t_v3_m, h_v3_m, eta_v3_m = simulate_meyerhofer(
        h0=h0,
        rho=rho,
        rpm=rpm,
        eta0=eta0,
        B=0.0,
        E=0.0,
        h_dry=0.0,
        t_end=t_end,
        dt=dt
    )
    max_diff = np.max(np.abs(h_v3_e - h_v3_m)) * 1e6

    val_rows.append({
        "Limit": "E → 0, B → 0",
        "Input condition": "E = 0, B = 0",
        "Expected result": "Meyerhofer → EBP",
        "Simulator result": f"Max difference = {max_diff:.6f} μm"
    })

    val_df = pd.DataFrame(val_rows)
    st.dataframe(val_df)


# =====================================================
# Tab 4
# =====================================================

with tab4:
    st.subheader("Challenge Mode: Find (ω, η₀) Combinations Satisfying Uniformity Spec")

    if st.button("Run Challenge Search"):
        search_df, success_df = challenge_search(
            spec=spec,
            rpm_min=rpm_min,
            rpm_max=rpm_max,
            eta_min=eta_min,
            eta_max=eta_max,
            h0=h0,
            rho=rho,
            B=B,
            E=E,
            h_dry=h_dry,
            R=R,
            t_end=t_end,
            dt=dt,
            Nr=Nr,
            edge_bead_strength=edge_bead_strength,
            edge_bead_width_ratio=edge_bead_width_ratio,
            leveling_coeff=leveling_coeff
        )

        st.subheader("All Search Results")
        st.dataframe(search_df)

        if len(success_df) == 0:
            st.error("No combination satisfied the uniformity specification.")
        else:
            best = success_df.iloc[0]
            st.success("Valid combinations were found.")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Best RPM", f"{best['RPM']:.0f}")
            c2.metric("Best ω", f"{best['omega_rad_s']:.2f} rad/s")
            c3.metric("Best η₀", f"{best['eta0_Pa_s']:.4f} Pa·s")
            c4.metric("Best Uniformity", f"±{best['final_uniformity_percent']:.3f} %")

            st.subheader("Successful Combinations")
            st.dataframe(success_df)

            fig, ax = plt.subplots(figsize=(10, 5))
            search_plot_df = search_df.reset_index()
            ax.plot(
                search_plot_df["index"],
                search_plot_df["final_uniformity_percent"],
                color="cyan",
                linewidth=2,
                marker="o",
                label="Uniformity"
            )
            ax.axhline(spec, color="red", linestyle="--", linewidth=2, label=f"Spec ±{spec:.2f}%")
            ax.set_xlabel("Search Case Index")
            ax.set_ylabel("Final Uniformity ± [%]")
            ax.set_title("Uniformity over RPM-η₀ Search Cases")
            ax.grid(True, alpha=0.3)
            ax.legend()
            st.pyplot(fig)

            fig, ax = plt.subplots(figsize=(9, 6))
            sc = ax.scatter(
                search_df["RPM"],
                search_df["eta0_Pa_s"],
                c=search_df["final_uniformity_percent"],
                cmap="viridis_r",
                s=80,
                edgecolors="black"
            )
            ax.set_xlabel("RPM")
            ax.set_ylabel("Initial Viscosity η₀ [Pa·s]")
            ax.set_title("RPM-η₀ Uniformity Map")
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label("Final Uniformity ± [%]")
            st.pyplot(fig)


# =====================================================
# Tab 5
# =====================================================

with tab5:
    st.subheader("Design Exploration Mode")

    design_df = pd.DataFrame({
        "Parameter": [
            "RPM",
            "Angular velocity ω [rad/s]",
            "Initial viscosity η₀ [Pa·s]",
            "Initial thickness h₀ [μm]",
            "Evaporation rate E [μm/s]",
            "Viscosity growth rate B [1/s]",
            "Gel criterion",
            "Gel viscosity η_gel [Pa·s]",
            "t_gel prediction [s]",
            "Wafer radius R [cm]",
            "Edge bead strength",
            "Edge bead width ratio",
            "Radial leveling coefficient",
            "Leveling rate [1/s]",
            "Final radial uniformity [%]"
        ],
        "Value": [
            rpm,
            rpm_to_omega(rpm),
            eta0,
            h0_um,
            E_um_s,
            B,
            f"{gel_factor:.0f} × η₀",
            eta_gel,
            "Not reached" if t_gel is None or t_gel > t_end else round(t_gel, 3),
            R_cm,
            edge_bead_strength,
            edge_bead_width_ratio,
            leveling_coeff,
            leveling_rate,
            final_uniformity
        ]
    })

    st.dataframe(design_df)

    if final_uniformity <= spec:
        st.success(f"This design satisfies the ±{spec:.2f}% uniformity specification.")
    else:
        st.warning(f"This design does not satisfy the ±{spec:.2f}% uniformity specification.")


# =====================================================
# Tab 6
# =====================================================

with tab6:
    st.subheader("Process-design Insight")

    st.markdown(
        f"""
### What the simulator teaches

1. **EBP model** predicts thinning caused only by centrifugal outflow.
2. **Meyerhofer-type model** includes solvent evaporation and viscosity growth, so it gives a more realistic late-stage thinning trend.
3. **RPM increase** strengthens radial spreading and improves final radial uniformity.
4. **Initial viscosity increase** suppresses radial flow and worsens uniformity.
5. **Gel time** is defined as the time when η(t) = 20η₀. If t_gel is larger than the process time, flow can still continue during coating.
6. **Challenge Mode** identifies the process window satisfying the ±{spec:.2f}% uniformity specification.

### Recommendation to a fab engineer

- Use sufficiently high RPM to improve radial spreading.
- Avoid excessively high η₀ because high viscosity prevents leveling.
- Control evaporation so that the film does not freeze before leveling.
- Reduce edge bead by optimizing dispense volume, acceleration ramp, and edge bead removal.
- Select process conditions satisfying:

Uniformity = (h_max - h_min) / (2 h_avg) × 100 ≤ ±{spec:.2f}%
        """
    )


# =====================================================
# Tab 7
# =====================================================

with tab7:
    st.subheader("Simulation Data")

    st.dataframe(radial_data)

    st.markdown("### Final Radial Profile Data")

    final_df = pd.DataFrame({
        "r_cm": r * 100.0,
        "final_h_um": final_profile * 1e6
    })

    st.dataframe(final_df)
