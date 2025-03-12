# import necessary libraries
import numpy as np


def compute_c_t(a, apply_glauert_correction=False):
    """
    This function calculates the thrust coefficient as a function of induction factor 'a'
    'glauert' defines if the Glauert correction for heavily loaded rotors should be used; default value is false
    """
    c_t = 4 * a * (1 - a)
    if apply_glauert_correction:
        CT1 = 1.816
        a1 = 1 - np.sqrt(CT1) / 2
        if type(a) is np.float64:
            if a > a1:
                c_t = CT1 - 4 * (np.sqrt(CT1) - 1) * (1 - a)
        else:
            c_t[a > a1] = CT1 - 4 * (np.sqrt(CT1) - 1) * (1 - a[a > a1])

    return c_t


def compute_axial_induction(c_t):
    """
    This function calculates the induction factor 'a' as a function of thrust coefficient CT
    including Glauert's correction
    """
    a = np.zeros(np.shape(c_t))
    CT1 = 1.816
    CT2 = 2 * np.sqrt(CT1) - CT1
    a[c_t >= CT2] = 1 + (c_t[c_t >= CT2] - CT1) / (4 * (np.sqrt(CT1) - 1))
    a[c_t < CT2] = 0.5 - 0.5 * np.sqrt(1 - c_t[c_t < CT2])
    return a


def prandtl_tip_root_correction(r_R, root_radius_over_R, tip_radius_over_R, tip_speed_ratio, blades_number, axial_induction):
    """
    This function calculates the combined tip and root Prandtl correction at a given radial position 'r_R' (non-dimensioned by rotor radius),
    given a root and tip radius (also non-dimensioned), a tip speed ratio TSR, the number of blades NBlades and the axial induction factor
    """
    temp1 = -blades_number / 2 * (tip_radius_over_R - r_R) / r_R * np.sqrt(
        1 + ((tip_speed_ratio * r_R) ** 2) / ((1 - axial_induction) ** 2))
    f_tip = np.array(2 / np.pi * np.arccos(np.exp(temp1)))
    f_tip[np.isnan(f_tip)] = 0
    temp1 = blades_number / 2 * (root_radius_over_R - r_R) / r_R * np.sqrt(
        1 + ((tip_speed_ratio * r_R) ** 2) / ((1 - axial_induction) ** 2))
    f_root = np.array(2 / np.pi * np.arccos(np.exp(temp1)))
    f_root[np.isnan(f_root)] = 0
    return f_root * f_tip, f_tip, f_root


def load_blade_element(v_normal, v_tangential, chord, twist, polar_alpha, polar_cl, polar_cd):
    """
    calculates the load in the blade element
    """
    v_magnitude_squared = v_normal ** 2 + v_tangential ** 2
    inflow_angle = np.arctan2(v_normal, v_tangential)
    alpha = inflow_angle * 180 / np.pi - twist
    if np.any(alpha > max(polar_alpha)) or np.any(alpha < min(polar_alpha)):
        raise Exception('Angle of attack out of range')

    cl = np.interp(alpha, polar_alpha, polar_cl)
    cd = np.interp(alpha, polar_alpha, polar_cd)
    lift = 0.5 * v_magnitude_squared * cl * chord
    drag = 0.5 * v_magnitude_squared * cd * chord
    normal_force = lift * np.cos(inflow_angle) + drag * np.sin(inflow_angle)
    tangential_force = lift * np.sin(inflow_angle) - drag * np.cos(inflow_angle)
    gamma = 0.5 * np.sqrt(v_magnitude_squared) * cl * chord
    return normal_force, tangential_force, gamma, alpha, inflow_angle


def BEM_cycle(u_inf, r_over_R, root_radius_over_R, tip_radius_over_R, omega, rotor_R, blades_number,
              chord_distribution, twist_distribution, yaw_angle, tip_speed_ratio, polar_alpha, polar_cl, polar_cd,
              psi_vec=[], prandtl_correction=True):

    if (not (prandtl_correction) and (yaw_angle != 0)):
        raise Exception(
            'The study of the effect of Prandtl correction is considered only for the non yawed case')

    if (len(psi_vec) == 0) and (yaw_angle != 0):
        raise Exception('The yawed case should have a psi vector')

    n_annuli = len(r_over_R) - 1
    if len(psi_vec) == 0:
        n_psi = 1
    else:
        n_psi = len(psi_vec)

    max_iterations = 100
    results = {}

    results = {
        'a': np.zeros([n_annuli, n_psi], dtype=float),
        'a_line': np.zeros([n_annuli, n_psi], dtype=float),
        'r_over_R': np.zeros([n_annuli, n_psi], dtype=float),
        'normal_force': np.zeros([n_annuli, n_psi], dtype=float),
        'tangential_force': np.zeros([n_annuli, n_psi], dtype=float),
        'gamma': np.zeros([n_annuli, n_psi], dtype=float),
        'alpha': np.zeros([n_annuli, n_psi], dtype=float),
        'inflow_angle': np.zeros([n_annuli, n_psi], dtype=float),
        'c_thrust': np.zeros([n_annuli, n_psi], dtype=float),
        'c_torque': np.zeros([n_annuli, n_psi], dtype=float),
        'c_power': np.zeros([n_annuli, n_psi], dtype=float),
        'psi': 0 if yaw_angle == 0 else psi_vec,
        'c_thrust_iterations': np.zeros([n_annuli, n_psi, max_iterations], dtype=float)
    }
    centroids = (r_over_R[1:] + r_over_R[:-1]) / 2
    for i in range(len(r_over_R) - 1):
        chord = np.interp(
            (r_over_R[i] + r_over_R[i + 1]) / 2, r_over_R, chord_distribution)
        twist = np.interp(
            (r_over_R[i] + r_over_R[i + 1]) / 2, r_over_R, twist_distribution)
        a, a_line, normal_force, tangential_force, gamma, alpha, inflow_angle, c_t, c_q, c_p, c_t_iterations, n_iterations = solve_stream_tube(
            u_inf, r_over_R[i], r_over_R[i + 1], root_radius_over_R, tip_radius_over_R, omega, rotor_R,
            blades_number, chord, twist, yaw_angle, tip_speed_ratio, polar_alpha, polar_cl, polar_cd, max_iterations, psi_vec, prandtl_correction)

        results['a'][i, :] = a
        results['a_line'][i, :] = a_line
        results['r_over_R'][i, :] = centroids[i]
        results['normal_force'][i, :] = normal_force
        results['tangential_force'][i, :] = tangential_force
        results['gamma'][i, :] = gamma
        results['alpha'][i, :] = alpha
        results['inflow_angle'][i, :] = inflow_angle
        results['c_thrust'][i, :] = c_t
        results['c_torque'][i, :] = c_q
        results['c_power'][i, :] = c_p
        results['c_thrust_iterations'][i, :, :] = c_t_iterations

    return results


def solve_stream_tube(u_inf, r1_over_R, r2_over_R, root_radius_over_R, tip_radius_over_R, omega, rotor_R, blades_number,
                      chord, twist, yaw_angle, tip_speed_ratio, polar_alpha, polar_cl, polar_cd, max_iterations, psi_vec=[], prandtl_correction=True):
    """
    solve balance of momentum between blade element load and loading in the stream-tube
    input variables:
    u_inf - wind speed at infinity
    r1_R,r2_R - edges of blade element, in fraction of Radius ;
    root_radius_over_R, tip_radius_over_R - location of blade root and tip, in fraction of Radius ;
    Radius is the rotor radius
    Omega -rotational velocity
    NBlades - number of blades in rotor
    """
    area = np.pi * ((r2_over_R * rotor_R) ** 2 -
                    (r1_over_R * rotor_R) ** 2)  # area annulus
    r_over_R = (r1_over_R + r2_over_R) / 2  # centroid
    dr = rotor_R * (r2_over_R - r1_over_R)

    # initialize variables
    normal_force = np.nan
    tangential_force = np.nan
    gamma = np.nan
    a = 0.0  # axial induction
    a_line = 0.0  # tangential induction factor
    error_iterations = 0.00001  # error limit for iteration process, in absolute value of induction
    c_t_iterations = np.empty(max_iterations)

    for i in range(max_iterations):
        u_rotor = u_inf * (1 - a)  # axial velocity at rotor
        u_tangential = (1 + a_line) * omega * r_over_R * rotor_R  # tangential velocity at rotor

        # calculate loads in blade segment in 2D (N/m)
        normal_force, tangential_force, gamma, alpha, inflow_angle = load_blade_element(
            u_rotor, u_tangential, chord, twist, polar_alpha, polar_cl, polar_cd)
        load_3d_axial = normal_force * dr * blades_number  # 3D force in axial direction
        # 3D force in azimuthal/tangential direction
        load_3d_tangential = normal_force * dr * blades_number
        c_t = load_3d_axial / (0.5 * area * u_inf ** 2)

        # calculate new axial induction, accounting for Glauert's correction
        a_new = compute_axial_induction(c_t)

        if prandtl_correction:
            # correct new axial induction with Prandtl's correction
            prandtl, prandtl_tip, prandtl_root = prandtl_tip_root_correction(
                r_over_R, root_radius_over_R, tip_radius_over_R, omega * rotor_R / u_inf, blades_number, a_new)
            if prandtl < 0.0001:
                prandtl = 0.0001  # avoid divide by zero
        else:
            prandtl = 1.0

        a_corrected = a_new / prandtl  # correct estimate of axial induction
        # for improving convergence, weigh current and previous iteration of axial induction
        a = 0.75 * a + 0.25 * a_corrected
        c_t_iterations[i] = compute_c_t(a_corrected, apply_glauert_correction=True)

        # calculate azimuthal induction
        a_line_new = tangential_force * blades_number / \
            (2 * np.pi * u_inf * (1 - a) * omega * 2 * (r_over_R * rotor_R) ** 2)
        a_line_corrected = a_line_new / prandtl
        a_line = 0.75 * a_line + 0.25 * a_line_corrected

        # test convergence of solution, by checking convergence of axial induction
        if np.abs(a - a_corrected) < error_iterations and np.abs(a_line - a_line_corrected) < error_iterations:
            break
    c_t = compute_c_t(a_corrected, apply_glauert_correction=True)
    c_q = 4 * a_line_corrected * (1 - a_corrected) * r_over_R * tip_speed_ratio
    c_p = c_t * (1 - a_corrected)
    inflow_angle = np.degrees(inflow_angle)
    iterations = i + 1

    if yaw_angle != 0:
        yaw_angle = np.radians(yaw_angle)
        dpsi = psi_vec[1:] - psi_vec[:-1]
        dpsi = np.append(dpsi, dpsi[-1])

        wake_skew_angle = (0.6*a_new + 1) * yaw_angle
        a_tot = a_new * (1 + 2 * np.tan(wake_skew_angle/2) * r_over_R * np.sin(psi_vec))

        if prandtl_correction:
            prandtl, prandtl_tip, prandtl_root = prandtl_tip_root_correction(
                r_over_R, root_radius_over_R, tip_radius_over_R, omega * rotor_R / u_inf, blades_number, a_tot)
            prandtl[prandtl < 0.0001] = 0.0001
        else:
            prandtl = np.ones_like(a_tot)

        a_corrected = a_tot / prandtl
        a_line_corrected = a_line_new / prandtl + 1/r_over_R * 1/tip_speed_ratio * np.sin(yaw_angle * np.sin(psi_vec))

        c_t = 4 * a_corrected * np.sqrt((1 - a_corrected*(2*np.cos(yaw_angle) - a_corrected)))
        c_p = c_t * (np.cos(yaw_angle) - a_corrected)
        c_q = 4 * a_line_corrected[i] * (np.cos(yaw_angle) - a_corrected[i]) * r_over_R * tip_speed_ratio * (
            np.cos(psi_vec)**2 + (np.cos(wake_skew_angle)**2) * (np.sin(psi_vec))**2)

        v_tan = omega*r_over_R*rotor_R*(1 + a_line_corrected)
        v_norm = u_inf*(np.cos(yaw_angle) - a_corrected)
        normal_force, tangential_force, gamma, alpha, inflow_angle = load_blade_element(
            v_norm, v_tan, chord, twist, polar_alpha, polar_cl, polar_cd)
        inflow_angle = np.degrees(inflow_angle)

    normal_force = normal_force / (0.5 * u_inf**2 * rotor_R)
    tangential_force = normal_force / (0.5 * u_inf**2 * rotor_R)
    gamma = gamma / (np.pi * u_inf**2 / (blades_number * omega))

    return a_corrected, a_line_corrected, normal_force, tangential_force, gamma, alpha, inflow_angle, c_t, c_q, c_p, c_t_iterations, iterations
