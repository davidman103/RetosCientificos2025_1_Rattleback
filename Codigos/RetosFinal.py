import numpy as np
from stl import mesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.linalg import lstsq # Para mínimos cuadrados
from numpy.polynomial import Polynomial # Para resolver polinomios

# --- Constantes Definidas
BODY_STL_FILE_PATH = 'C:/Users/deivy/Downloads/Archivos_stl/Rattleback - 4974/RATTLEBACKCORREGIDOORIENTADOCCW.stl'
CONTACT_SURFACE_STL_FILE_PATH = 'C:/Users/deivy/Downloads/Archivos_stl/superficieRattleback_aproximada_escalada_final.stl'
DENSITY_G_PER_CM3 = 1.24
SCALE_FACTOR_STL_TO_WORKING_UNITS = 0.1 # Asumiendo STL en mm, trabajando en cm
WORKING_UNIT_NAME = "cm"
INITIAL_Z_TRANSLATION_WU = 1.0 # Traslación inicial de +1 cm

# Parámetros para la selección de puntos para el ajuste de superficie (en el marco CoM)
FIT_REGION_XY_RADIUS_CoM_WU = 5.0  # Radio en el plano XY del CoM alrededor de (xc,yc) del contacto
FIT_REGION_Z_DEVIATION_CoM_WU = 0.5 # +/- desviación en Z (del CoM, Z abajo) alrededor de 'a_eff' para incluir puntos

# Parámetros para el análisis dinámico
GRAVITY_WU = 981 # cm/s^2 (aproximado)
SPIN_RATES_N_RAD_PER_S = np.linspace(0.5, 100, 50) # Rango de velocidades de giro n para analizar (rad/s)

# --- Funciones (sin cambios en las funciones auxiliares, solo en main y posiblemente en plot) ---
def load_stl(file_path):
    if file_path is None: return None
    try:
        return mesh.Mesh.from_file(file_path).vectors
    except Exception as e:
        print(f"Error al cargar '{file_path}': {e}"); return None

def print_model_dimensions(triangles, unit_name="unidades", model_name="Modelo"):
    if triangles is None or triangles.size == 0:
        print(f"{model_name}: No hay datos."); return
    all_vertices = triangles.reshape(-1, 3)
    min_c, max_c = np.min(all_vertices, axis=0), np.max(all_vertices, axis=0)
    print(f"   {model_name} - Mín: {min_c} {unit_name}, Máx: {max_c} {unit_name}, Dim: {max_c - min_c} {unit_name}")

def apply_scale_factor(triangles, scale_factor):
    if triangles is None: return None
    return triangles * scale_factor

def initial_translate_z(triangles, z_translation):
    if triangles is None: return None
    triangles_translated = triangles.copy()
    triangles_translated[:, :, 2] += z_translation
    return triangles_translated

def calculate_volume_and_center_of_mass(triangles):
    if triangles is None or triangles.size == 0: return 0.0, np.zeros(3)
    total_volume = 0.0
    cm_num = np.zeros(3, dtype=float)
    for v0, v1, v2 in triangles:
        vol_tet = np.dot(v0, np.cross(v1, v2)) / 6.0
        total_volume += vol_tet
        cm_num += vol_tet * (v0 + v1 + v2) / 4.0
    if abs(total_volume) < 1e-12:
        print("ADVERTENCIA: Volumen del cuerpo principal cercano a cero.")
        return 0.0, np.mean(triangles.reshape(-1,3), axis=0) if triangles.size > 0 else np.zeros(3)
    return abs(total_volume), cm_num / total_volume

def calculate_inertia_tensor(triangles, center_of_mass_obj, density):
    if triangles is None or triangles.size == 0: return np.zeros((3,3))
    I = np.zeros((3, 3))
    triangles_shifted = triangles - center_of_mass_obj
    for v0_s, v1_s, v2_s in triangles_shifted:
        vol_tet = np.dot(v0_s, np.cross(v1_s, v2_s)) / 6.0
        mass_tet = density * abs(vol_tet)
        if abs(vol_tet) < 1e-12: continue
        rc_tet = (v0_s + v1_s + v2_s) / 4.0
        x, y, z = rc_tet
        I[0,0] += mass_tet*(y*y + z*z)
        I[1,1] += mass_tet*(x*x + z*z)
        I[2,2] += mass_tet*(x*x + y*y)
        I[0,1] -= mass_tet*x*y; I[1,0] = I[0,1]
        I[0,2] -= mass_tet*x*z; I[2,0] = I[0,2]
        I[1,2] -= mass_tet*y*z; I[2,1] = I[1,2]
    return I

def find_lowest_z_vertex(vertices_list): # Esta función podría no ser necesaria para el nuevo enfoque
    if vertices_list is None or vertices_list.shape[0] == 0:
        return None
    min_z_idx = np.argmin(vertices_list[:, 2])
    return vertices_list[min_z_idx]

def fit_quadratic_surface(points_xyz): # points_xyz tiene columnas (x_local_fit, y_local_fit, z_target_fit)
    if points_xyz is None or points_xyz.shape[0] < 6:
        print("No hay suficientes puntos para un ajuste cuadrático robusto.")
        return np.full(6, np.nan)
    x = points_xyz[:, 0]
    y = points_xyz[:, 1]
    z = points_xyz[:, 2]
    A_matrix = np.vstack([np.ones_like(x), x, y, x**2, y**2, x*y]).T
    try:
        coeffs, residuals, rank, singular_values = lstsq(A_matrix, z)
        if rank < A_matrix.shape[1]:
            print(f"ADVERTENCIA: La matriz de diseño tiene rango deficiente ({rank} < {A_matrix.shape[1]}).")
    except np.linalg.LinAlgError as e:
        print(f"Error en mínimos cuadrados: {e}")
        return np.full(6, np.nan)
    return coeffs

def display_fitted_polynomial_and_taylor(fit_coeffs, a_eff_direct=None):
    if fit_coeffs is None or np.any(np.isnan(fit_coeffs)):
        print("No se pueden mostrar los coeficientes del polinomio debido a un error previo.")
        return
    c0, c1, c2, c3, c4, c5 = fit_coeffs
    terms = []
    if not np.isclose(c0, 0) or (len(terms) == 0 and np.allclose(fit_coeffs[1:],0) ):
        terms.append(f"{c0:.4f}")
    if not np.isclose(c1, 0): terms.append(f"{c1:+.4f}x'")
    if not np.isclose(c2, 0): terms.append(f"{c2:+.4f}y'")
    if not np.isclose(c3, 0): terms.append(f"{c3:+.4f}x'^2")
    if not np.isclose(c4, 0): terms.append(f"{c4:+.4f}y'^2")
    if not np.isclose(c5, 0): terms.append(f"{c5:+.4f}x'y'")
    if not terms: polynomial_str = "Z_CoM(x', y') = 0.0000" # Ajustado para el nuevo contexto
    else:
        polynomial_str = "Z_CoM(x', y') = " + " ".join(terms) # Ajustado para el nuevo contexto
        if polynomial_str.startswith("Z_CoM(x', y') =  +"):
            polynomial_str = "Z_CoM(x', y') = " + polynomial_str[len("Z_CoM(x', y') =  +"):].strip()
        elif polynomial_str.startswith("Z_CoM(x', y') = +"):
            polynomial_str = "Z_CoM(x', y') = " + polynomial_str[len("Z_CoM(x', y') = +"):].strip()

    print("\n--- Polinomio Cuadrático Ajustado (Marco CoM, Z abajo, X'Y' rel. a contacto) ---")
    print(polynomial_str)
    print(f"\nCoeficientes brutos del ajuste (c0 a c5): {fit_coeffs}")
    if a_eff_direct is not None:
        print(f"   c0 (debería ser aprox. a_eff): {c0:.4f} (comparar con a_eff directo: {a_eff_direct:.4f})")
    else:
        print(f"  c0 (intercepto Z en el origen x'y'): {c0:.4f}")
    print(f"  c1 (pendiente en x' en el origen x'y'): {c1:.4f} (debería ser pequeño)")
    print(f"  c2 (pendiente en y' en el origen x'y'): {c2:.4f} (debería ser pequeño)")
    # Interpretación de c3,c4,c5 para p,s,q se hace en main

def plot_3d_visualization(triangles_body_plot_global, cm_body_global, principal_axes_body_plot_rows_global, unit_name,
                            contact_pt_highlight_global=None,
                            points_for_fit_input_coords=None, # (x', y', z_CoM)
                            fit_coeffs_for_plot=None,
                            xc_contact_CoM=None, yc_contact_CoM=None): # Necesarios para transformar puntos de ajuste para visualización global
    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.set_title(f"Rattleback: CM, Ejes (Global, {unit_name})", fontsize=10)

    if triangles_body_plot_global is not None and triangles_body_plot_global.size > 0:
        ax1.add_collection3d(Poly3DCollection(triangles_body_plot_global, facecolors='cornflowerblue', linewidths=0.3, edgecolors='grey', alpha=0.4, label="Cuerpo Principal"))
    ax1.scatter(cm_body_global[0], cm_body_global[1], cm_body_global[2], c='red', s=80, label="CM Global", depthshade=False)

    colors = ['green', 'purple', 'orange']
    labels_axes = ['Eje Principal $I_1$ (A o B)', 'Eje Principal $I_2$ (B o A)', 'Eje Principal $I_3$ (C)']
    model_span_pts = triangles_body_plot_global.reshape(-1,3) if triangles_body_plot_global is not None and triangles_body_plot_global.size>0 else np.array([[0,0,0],[1,1,1]])
    model_span = np.max(np.ptp(model_span_pts, axis=0))
    axis_length = model_span * 0.6

    if principal_axes_body_plot_rows_global.shape == (3,3):
        for i, axis_vec in enumerate(principal_axes_body_plot_rows_global):
            ax1.quiver(cm_body_global[0], cm_body_global[1], cm_body_global[2],
                       axis_vec[0], axis_vec[1], axis_vec[2],
                       color=colors[i], length=axis_length, normalize=True, label=labels_axes[i], arrow_length_ratio=0.1)

    if contact_pt_highlight_global is not None:
        ax1.scatter(contact_pt_highlight_global[0], contact_pt_highlight_global[1], contact_pt_highlight_global[2],
                    c='magenta', s=120, marker='P', label="Punto de Contacto (Global Est.)", depthshade=False, edgecolor='black')

    if points_for_fit_input_coords is not None and points_for_fit_input_coords.shape[0] > 0 and \
            xc_contact_CoM is not None and yc_contact_CoM is not None:
        points_for_plot_global = []
        for p_fit in points_for_fit_input_coords:
            x_prime, y_prime, z_val_CoM_down = p_fit
            # Transformar de (x', y', z_CoM_down) a global
            x_CoM_frame = x_prime + xc_contact_CoM
            y_CoM_frame = y_prime + yc_contact_CoM
            z_CoM_frame_Zup = -z_val_CoM_down # Z original del CoM (arriba)
            global_pt = np.array([x_CoM_frame, y_CoM_frame, z_CoM_frame_Zup]) + cm_body_global
            points_for_plot_global.append(global_pt)
        points_for_plot_global = np.array(points_for_plot_global)
        ax1.scatter(points_for_plot_global[:,0], points_for_plot_global[:,1], points_for_plot_global[:,2],
                    c='darkorange', s=10, label="Puntos de Ajuste (Global)", alpha=0.7, depthshade=False)

    plot_center = cm_body_global
    max_ptp = np.max(np.ptp(model_span_pts, axis=0)) if model_span_pts.size > 3 else 10.0
    plot_range_val = max_ptp * 0.8
    if plot_range_val < 1e-6: plot_range_val = 5.0
    ax1.set_xlim(plot_center[0] - plot_range_val, plot_center[0] + plot_range_val)
    ax1.set_ylim(plot_center[1] - plot_range_val, plot_center[1] + plot_range_val)
    ax1.set_zlim(plot_center[2] - plot_range_val, plot_center[2] + plot_range_val) # Z global original
    ax1.set_xlabel(f"X global ({unit_name})"); ax1.set_ylabel(f"Y global ({unit_name})"); ax1.set_zlabel(f"Z global ({unit_name})")
    ax1.legend(fontsize=8); ax1.view_init(elev=25, azim=-135)

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.set_title("Sup. Ajustada (Marco CoM, Z+ abajo, X'Y' rel. contacto)", fontsize=10)
    if points_for_fit_input_coords is not None and points_for_fit_input_coords.shape[0] > 0 and \
       fit_coeffs_for_plot is not None and not np.any(np.isnan(fit_coeffs_for_plot)):

        x_prime_data = points_for_fit_input_coords[:,0]
        y_prime_data = points_for_fit_input_coords[:,1]
        z_CoM_data = points_for_fit_input_coords[:,2] # Estos son Z_CoM (Z abajo)

        x_loc_min, x_loc_max = np.min(x_prime_data), np.max(x_prime_data)
        y_loc_min, y_loc_max = np.min(y_prime_data), np.max(y_prime_data)
        padding_x = (x_loc_max - x_loc_min) * 0.1 if (x_loc_max - x_loc_min) > 1e-6 else 0.1
        padding_y = (y_loc_max - y_loc_min) * 0.1 if (y_loc_max - y_loc_min) > 1e-6 else 0.1

        X_surf_loc_plot = np.linspace(x_loc_min - padding_x, x_loc_max + padding_x, 30)
        Y_surf_loc_plot = np.linspace(y_loc_min - padding_y, y_loc_max + padding_y, 30)
        X_mesh, Y_mesh = np.meshgrid(X_surf_loc_plot, Y_surf_loc_plot)

        c0, c1, c2, c3, c4, c5 = fit_coeffs_for_plot
        Z_fitted_plot = c0 + c1*X_mesh + c2*Y_mesh + c3*X_mesh**2 + c4*Y_mesh**2 + c5*X_mesh*Y_mesh

        ax2.plot_surface(X_mesh, Y_mesh, Z_fitted_plot,
                         color='cyan', alpha=0.7, rstride=1, cstride=1, edgecolor='k', linewidth=0.2)
        ax2.scatter(x_prime_data, y_prime_data, z_CoM_data,
                    c='red', s=10, label="Puntos de Ajuste (x',y',Z_CoM)", depthshade=False, alpha=0.8)
        ax2.scatter([0], [0], [c0], color='black', s=50, marker='o', label=f"Origen Ajuste (0,0,{c0:.3f}) Z_CoM")

        ax2.set_xlabel(f"X' ({unit_name}, rel. a contacto en CoM)")
        ax2.set_ylabel(f"Y' ({unit_name}, rel. a contacto en CoM)")
        ax2.set_zlabel(f"Z_CoM ({unit_name}, desde CoM, Z+ abajo)")

        # Ajustar límites de Z para la vista local
        z_min_plot, z_max_plot = np.min(Z_fitted_plot), np.max(Z_fitted_plot)
        z_range_plot = z_max_plot - z_min_plot
        if z_range_plot < 1e-6 : z_range_plot = 0.1
        ax2.set_zlim(z_min_plot - 0.1 * z_range_plot, z_max_plot + 0.1 * z_range_plot)
        ax2.legend(fontsize=8); ax2.view_init(elev=20, azim=-60)
    else:
        ax2.text(0.5, 0.5, 0.5, "No hay datos para graficar sup. ajustada.",
                 horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
    plt.tight_layout(pad=2.0); plt.suptitle("Análisis del Rattleback (Marco CoM para Ajuste)", fontsize=14, y=0.98); fig.subplots_adjust(top=0.92)
    plt.show()

def analyze_characteristic_equation(alpha_B, beta_B, gamma_B, p_geom, q_geom, s_geom, a_eff, n_spin, g_const):
    if abs(n_spin) < 1e-6 or np.isnan(a_eff) or a_eff <= 1e-6 : # Añadida verificación de a_eff
        return np.full(4, np.nan, dtype=complex)
    g_a = g_const / a_eff
    n2 = n_spin**2; n3 = n_spin**3; n4 = n_spin**4

    A_curly = alpha_B * beta_B * (p_geom * s_geom - q_geom**2)
    B_curly_n_factor = (alpha_B - beta_B) * q_geom
    C_curly = (1 - p_geom * (alpha_B - gamma_B) - s_geom * (beta_B - gamma_B) +
               (p_geom * s_geom - q_geom**2) * (alpha_B - gamma_B) * (beta_B - gamma_B))
    D_term = n2 * g_a * (2 - (1 + alpha_B + beta_B - gamma_B) * (p_geom + s_geom) +
                       2 * (alpha_B + beta_B - gamma_B) * (p_geom * s_geom - q_geom**2))
    E_term = g_a**2 * ((1 - p_geom) * (1 - s_geom) - q_geom**2)

    K4 = A_curly
    K3 = B_curly_n_factor * n_spin
    K2 = C_curly * n2 + A_curly * n2
    K1 = B_curly_n_factor * n3
    K0 = C_curly * n4 + D_term + E_term

    if np.any(np.isnan([K0,K1,K2,K3,K4])): # Verificar si algún coeficiente es NaN
        # print(f"Advertencia: NaN en coeficientes K para n={n_spin}. K0={K0} K1={K1} K2={K2} K3={K3} K4={K4}")
        return np.full(4, np.nan, dtype=complex)
    try:
        poly = Polynomial([K0, K1, K2, K3, K4])
        roots_sigma = poly.roots()
    except Exception as e:
        # print(f"Error al calcular raíces para n={n_spin}: {e}")
        return np.full(4, np.nan, dtype=complex)
    return roots_sigma

def main():
    print("Iniciando análisis del Rattleback (Ajuste en Marco CoM)...")
    triangles_body_orig = load_stl(BODY_STL_FILE_PATH)
    if triangles_body_orig is None:
        print("Error: No se pudo cargar el STL del cuerpo principal. Terminando.")
        return

    print(f"\nCuerpo Principal (STL 1): {BODY_STL_FILE_PATH}")
    triangles_body_wu_global = apply_scale_factor(triangles_body_orig, SCALE_FACTOR_STL_TO_WORKING_UNITS)
    triangles_body_wu_global = initial_translate_z(triangles_body_wu_global, INITIAL_Z_TRANSLATION_WU)
    print_model_dimensions(triangles_body_wu_global, WORKING_UNIT_NAME, "Cuerpo Principal (Global)")

    volume, cm_body_global = calculate_volume_and_center_of_mass(triangles_body_wu_global)
    if abs(volume) < 1e-9:
        print("Error: Volumen del cuerpo principal es cercano a cero.")
        return

    mass = DENSITY_G_PER_CM3 * volume
    # Tensor de inercia calculado con origen en el CM global (Z original del STL hacia arriba)
    inertia_tensor_cm_body = calculate_inertia_tensor(triangles_body_wu_global, cm_body_global, DENSITY_G_PER_CM3)

    eigenvalues, eigenvectors_as_cols = np.linalg.eigh(inertia_tensor_cm_body)
    principal_axes_body_rows_global = eigenvectors_as_cols.T # Ejes en el marco global

    print(f"\n--- Propiedades del Cuerpo Principal ---")
    print(f"Volumen: {volume:.4f} {WORKING_UNIT_NAME}^3, Masa: {mass:.4f} g")
    print(f"CM Global: {cm_body_global} {WORKING_UNIT_NAME}")
    print(f"Tensor de Inercia (respecto a CM Global):\n{inertia_tensor_cm_body}")
    # ... (asignación A,B,C como antes)
    z_axis_global_ref = np.array([0, 0, 1]) # Eje Z de referencia global
    projections_on_z = [abs(np.dot(axis, z_axis_global_ref)) for axis in principal_axes_body_rows_global]
    vertical_axis_idx = np.argmax(projections_on_z)
    C_inertia = eigenvalues[vertical_axis_idx]
    remaining_eigenvalues = np.delete(eigenvalues, vertical_axis_idx)
    A_inertia = np.max(remaining_eigenvalues); B_inertia = np.min(remaining_eigenvalues)
    print(f"\nMomentos de Inercia asignados (A>B): A={A_inertia:.4f}, B={B_inertia:.4f}, C={C_inertia:.4f} g·{WORKING_UNIT_NAME}²")
    if not A_inertia > B_inertia:
        print("ADVERTENCIA: No se cumple A > B.")


    # --- Procesamiento de la Superficie y Ajuste en Marco CoM ---
    print("\n--- Ajuste de Superficie en Marco CoM (Z positivo hacia abajo) ---")
    # Usar los vértices del cuerpo principal para el ajuste de superficie
    all_body_vertices_global = triangles_body_wu_global.reshape(-1, 3)
    unique_body_vertices_global = np.unique(all_body_vertices_global, axis=0)

    # 1. Vértices de superficie en Marco CoM del CUERPO, Z positivo hacia abajo
    vertices_CoM_frame_Zup = unique_body_vertices_global - cm_body_global
    vertices_CoM_frame_Zdown = vertices_CoM_frame_Zup.copy()
    vertices_CoM_frame_Zdown[:, 2] *= -1 # Z positivo es hacia abajo desde CoM

    # 2. Encontrar el punto de contacto en el marco CoM (Z abajo) -> será (xc, yc, a_eff)
    if vertices_CoM_frame_Zdown.shape[0] == 0:
        print("Error: No hay vértices en el marco CoM Z-down.")
        return
    contact_idx_CoM = np.argmax(vertices_CoM_frame_Zdown[:, 2])
    contact_pt_CoM_Zdown_actual = vertices_CoM_frame_Zdown[contact_idx_CoM]

    xc_contact_CoM = contact_pt_CoM_Zdown_actual[0]
    yc_contact_CoM = contact_pt_CoM_Zdown_actual[1]
    a_eff = contact_pt_CoM_Zdown_actual[2] # Este es 'a' de Bondi

    print(f"Punto de contacto (xc,yc) en XY del CoM: ({xc_contact_CoM:.4f}, {yc_contact_CoM:.4f})")
    print(f"Distancia 'a' (CoM a contacto, Z-abajo): {a_eff:.4f} {WORKING_UNIT_NAME}")

    if a_eff <= 1e-3: # 'a' debe ser una distancia positiva razonable
        print(f"Error: 'a_eff' ({a_eff:.4f}) es muy pequeño o no positivo. Revisar geometría o CoM.")
        # Considerar plotear aquí para depuración si es necesario.
        plot_3d_visualization(triangles_body_wu_global, cm_body_global, principal_axes_body_rows_global, WORKING_UNIT_NAME,
                            contact_pt_highlight_global=None, # Se podría transformar contact_pt_CoM_Zdown_actual a global para plotear
                            points_for_fit_input_coords=None, fit_coeffs_for_plot=None,
                            xc_contact_CoM=None, yc_contact_CoM=None)
        return

    # 3. Seleccionar puntos para el ajuste y transformarlos a (x', y', z_CoM)
    #    donde x' = x_CoM - xc_contact_CoM, y' = y_CoM - yc_contact_CoM
    points_for_fit_input = []
    for v_CoM_zd in vertices_CoM_frame_Zdown:
        x_prime = v_CoM_zd[0] - xc_contact_CoM
        y_prime = v_CoM_zd[1] - yc_contact_CoM
        z_val_CoM = v_CoM_zd[2] # Este es el Z que se va a ajustar, Z positivo hacia abajo

        # Filtrar puntos para el ajuste
        if np.sqrt(x_prime**2 + y_prime**2) < FIT_REGION_XY_RADIUS_CoM_WU and \
           abs(z_val_CoM - a_eff) < FIT_REGION_Z_DEVIATION_CoM_WU: # Puntos cercanos a 'a_eff'
            points_for_fit_input.append([x_prime, y_prime, z_val_CoM])
    points_for_fit_input = np.array(points_for_fit_input)
    print(f"Número de puntos seleccionados para el ajuste (Marco CoM): {points_for_fit_input.shape[0]}")

    fit_coeffs = None
    p_geom, q_geom, s_geom = np.nan, np.nan, np.nan
    contact_pt_global_for_plot = None # Para visualización

    if points_for_fit_input.shape[0] >= 6:
        fit_coeffs = fit_quadratic_surface(points_for_fit_input) # Ajusta Z_CoM = f(x', y')
        display_fitted_polynomial_and_taylor(fit_coeffs, a_eff_direct=a_eff)

        if fit_coeffs is not None and not np.any(np.isnan(fit_coeffs)):
            c0_fit, c1_fit, c2_fit, c3_fit, c4_fit, c5_fit = fit_coeffs
            # c0_fit debería ser aproximadamente a_eff

            # Parámetros geométricos de Bondi p,s,q
            # p = -2*a*C3, s = -2*a*C4, q = -a*C5
            # Para p,s > 0 (convexidad de Bondi), C3 y C4 deben ser < 0.
            p_geom = -2 * a_eff * c3_fit
            s_geom = -2 * a_eff * c4_fit
            q_geom = -a_eff * c5_fit

            print(f"\nParámetros geométricos de Bondi estimados (p, s, q):")
            print(f"  p = {p_geom:.4f} (Esperado > 0 para convexidad de Bondi)")
            print(f"  s = {s_geom:.4f} (Esperado > 0 para convexidad de Bondi)")
            print(f"  q = {q_geom:.4f} (No cero para asimetría)")

            # Verificar condiciones de Bondi (Eq 7 y 8)
            print("\nVerificación de condiciones geométricas de Bondi:")
            valid_geom_params = True
            if not (p_geom > 1e-6 and s_geom > 1e-6 and (p_geom * s_geom > q_geom**2)): # Pequeña tolerancia para >0
                print(f"  Condición de Convexidad (Eq 7: p>0,s>0,ps>q^2): NO CUMPLIDA (p={p_geom:.3f}, s={s_geom:.3f}, ps-q^2={(p_geom*s_geom - q_geom**2):.3f})")
                valid_geom_params = False
            else:
                print("  Condición de Convexidad (Eq 7): CUMPLIDA")

            if not (1 > p_geom and 1 > s_geom and ((1 - p_geom) * (1 - s_geom) > q_geom**2)):
                print(f"  Condición de Estabilidad en Reposo (Eq 8): NO CUMPLIDA (1-p={(1-p_geom):.3f}, 1-s={(1-s_geom):.3f}, (1-p)(1-s)-q^2={((1-p_geom)*(1-s_geom)-q_geom**2):.3f})")
            else:
                print("  Condición de Estabilidad en Reposo (Eq 8): CUMPLIDA")
    else:
        print("No hay suficientes puntos para el ajuste en marco CoM.")
        valid_geom_params = False # No se puede continuar sin p,s,q

    # Preparar punto de contacto para ploteo global
    # contact_pt_CoM_Zdown_actual es (xc_CoM, yc_CoM, a_eff)
    temp_contact_global = contact_pt_CoM_Zdown_actual.copy()
    temp_contact_global[2] *= -1 # Convertir a_eff a Z_up desde CoM
    contact_pt_global_for_plot = temp_contact_global + cm_body_global


    # --- Cálculo de Parámetros Inerciales de Bondi (alpha, beta, gamma) ---
    if not np.isnan(a_eff) and abs(a_eff) > 1e-3 and mass > 1e-6 :
        Ma2 = mass * a_eff**2
        alpha_B = (A_inertia + Ma2) / Ma2
        beta_B = (B_inertia + Ma2) / Ma2
        gamma_B = C_inertia / Ma2

        print(f"\n--- Parámetros Inerciales Adimensionales de Bondi ---")
        print(f"  alpha = {alpha_B:.4f}, beta  = {beta_B:.4f}, gamma = {gamma_B:.4f}")

        # --- Análisis de Estabilidad del Giro ---
        if valid_geom_params and not np.any(np.isnan([p_geom, q_geom, s_geom])): # Asegurarse que p,s,q son válidos
            print(f"\n--- Análisis de Estabilidad del Giro (Raíces sigma de Eq. Característica) ---")
            print(f"Usando: p_geom={p_geom:.3f}, q_geom={q_geom:.3f}, s_geom={s_geom:.3f}, a_eff={a_eff:.3f}")

            concordance_spin_direction = None
            print("\n n (rad/s) | Raíces sigma (real, imag)                                    | Re(max) | Concordancia? | Inversión?")
            print("-----------|--------------------------------------------------------------|---------|---------------|------------")

            for n_val in SPIN_RATES_N_RAD_PER_S:
                # Los p,s,q calculados p_geom, s_geom, q_geom ya son los de Bondi.
                # p_geom y s_geom deben ser > 0 para convexidad.
                # La ecuación característica (20) usa estos p,s,q directamente.
                roots_pos_n = analyze_characteristic_equation(alpha_B, beta_B, gamma_B,
                                                              p_geom, q_geom, s_geom, # Usar directamente
                                                              a_eff, n_val, GRAVITY_WU)

                re_parts_pos_n = np.real(roots_pos_n)
                max_re_pos_n = np.max(re_parts_pos_n) if not np.all(np.isnan(re_parts_pos_n)) else np.nan

                stable_pos_n = np.all(re_parts_pos_n < 1e-4)
                stable_neg_n = np.all(-re_parts_pos_n < 1e-4) # Raíces para -n son -sigma_pos_n

                concordance_status = "No"
                if stable_pos_n and not stable_neg_n: concordance_status = "Sí (+n)"; concordance_spin_direction = "+"
                elif not stable_pos_n and stable_neg_n: concordance_status = "Sí (-n)"; concordance_spin_direction = "-"
                elif stable_pos_n and stable_neg_n: concordance_status = "Ambas?"

                inversion_hint = "-"
                if not stable_pos_n and stable_neg_n : inversion_hint = "Hacia -n"
                elif stable_pos_n and not stable_neg_n: inversion_hint = "Hacia +n"

                root_str_parts = []
                if not np.all(np.isnan(roots_pos_n)):
                    for r_idx, r_val in enumerate(roots_pos_n):
                        root_str_parts.append(f"s{r_idx+1}=({r_val.real:.2e},{r_val.imag:.2e})")
                else: root_str_parts.append("NaN                                ") # Espacio para alinear

                print(f" {n_val:9.2f} | {' '.join(root_str_parts):<60} | {max_re_pos_n:7.2e} | {concordance_status:<13} | {inversion_hint}")

            if concordance_spin_direction:
                print(f"\nSe encontró 'concordancia': el sistema tiende a ser estable para giros en la dirección '{concordance_spin_direction}'.")
            else:
                print("\nNo se encontró 'concordancia' clara (o ambas direcciones inestables/estables).")
        else:
            print("\nNo se pueden realizar análisis de estabilidad sin parámetros geométricos p,q,s válidos.")
    else:
        print("\nNo se pueden calcular parámetros inerciales de Bondi sin 'a_eff' o 'mass' válidos.")

    # --- Visualización ---
    plot_3d_visualization(
        triangles_body_wu_global,
        cm_body_global,
        principal_axes_body_rows_global,
        WORKING_UNIT_NAME,
        contact_pt_highlight_global=contact_pt_global_for_plot, # Punto de contacto transformado a global
        points_for_fit_input_coords=points_for_fit_input if points_for_fit_input.shape[0] > 0 else None, # (x',y',z_CoM)
        fit_coeffs_for_plot=fit_coeffs,
        xc_contact_CoM=xc_contact_CoM if 'xc_contact_CoM' in locals() else None, # Pasar para transformación en plot
        yc_contact_CoM=yc_contact_CoM if 'yc_contact_CoM' in locals() else None  # Pasar para transformación en plot
    )
    print("\nAnálisis completado.")

if __name__ == "__main__":
    main()