# import necessary libraries
import numpy as np
import math
from mpmath import sec

def compute_c_t(a, apply_glauert_correction=False):
    """
    This function calculates the thrust coefficient as a function of induction factor 'a'
    'glauert' defines if the Glauert correction for heavily loaded rotors should be used; default value is false
    """
    c_t = 4 * a * (1 - a)
    if apply_glauert_correction:
        CT1 = 1.816
        a1 = 1 - np.sqrt(CT1) / 2
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
    cl = np.interp(alpha, polar_alpha, polar_cl)
    cd = np.interp(alpha, polar_alpha, polar_cd)
    lift = 0.5 * v_magnitude_squared * cl * chord
    drag = 0.5 * v_magnitude_squared * cd * chord
    normal_force = lift * np.cos(inflow_angle) + drag * np.sin(inflow_angle)
    tangential_force = lift * \
        np.sin(inflow_angle) - drag * np.cos(inflow_angle)
    gamma = 0.5 * np.sqrt(v_magnitude_squared) * cl * chord
    return normal_force, tangential_force, gamma, alpha, inflow_angle

def solve_stream_tube(u_inf, r1_over_R, r2_over_R, root_radius_over_R, tip_radius_over_R, omega, rotor_R, blades_number,
                      chord, twist, yaw_angle, tip_speed_ratio, polar_alpha, polar_cl, polar_cd, psi_vec = [], prandtl_correction = True):
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
    if (not(prandtl_correction) and (yaw_angle != 0)):
        raise Exception('The study of the effect of Prandtl correction is considered only for the non yawed case')
    
    if (len(psi_vec) == 0) and (yaw_angle != 0):
        raise Exception('The yawed case should have a psi vector')
    
    area = np.pi * ((r2_over_R * rotor_R) ** 2 - (r1_over_R * rotor_R) ** 2)  # area annulus
    r_over_R = (r1_over_R + r2_over_R) / 2  # centroid
    dr = rotor_R * (r2_over_R - r1_over_R)

    # initialize variables
    normal_force = np.nan
    tangential_force = np.nan
    gamma = np.nan
    a = 0.0  # axial induction
    a_line = 0.0  # tangential induction factor
    n_iterations = 100
    # error limit for iteration process, in absolute value of induction
    error_iterations = 0.00001
    c_t_iterations = np.empty(n_iterations)

    for i in range(n_iterations):
        u_rotor = u_inf * (1 - a)  # axial velocity at rotor
        u_tangential = (1 + a_line) * omega * r_over_R * rotor_R  # tangential velocity at rotor
        
        # calculate loads in blade segment in 2D (N/m)
        normal_force, tangential_force, gamma, alpha, inflow_angle = load_blade_element(
            u_rotor, u_tangential, chord, twist, polar_alpha, polar_cl, polar_cd)
        load_3d_axial = normal_force * dr * blades_number  # 3D force in axial direction
        load_3d_tangential = normal_force * dr * blades_number  # 3D force in azimuthal/tangential direction

        c_t = load_3d_axial / (0.5 * area * u_inf ** 2)

        # calculate new axial induction, accounting for Glauert's correction
        a_new = compute_axial_induction(c_t)

        if prandtl_correction:
            # correct new axial induction with Prandtl's correction
            prandtl, prandtl_tip, prandtl_root = prandtl_tip_root_correction(r_over_R, root_radius_over_R, tip_radius_over_R, omega * rotor_R / u_inf, blades_number, a_new)
            if prandtl < 0.0001: prandtl = 0.0001  # avoid divide by zero
        else:
            prandtl = 1.0

        a_corrected = a_new / prandtl  # correct estimate of axial induction
        # for improving convergence, weigh current and previous iteration of axial induction
        a = 0.75 * a + 0.25 * a_corrected
        c_t_iterations[i] = 4 * a_corrected * ( 1 - a_corrected)

        # calculate azimuthal induction
        a_line_new = tangential_force * blades_number / (2 * np.pi * u_inf * (1 - a) * omega * 2 * (r_over_R * rotor_R) ** 2)
        a_line_corrected = a_line_new / prandtl
        a_line = 0.75 * a_line + 0.25 * a_line_corrected

        # test convergence of solution, by checking convergence of axial induction
        if np.abs(a - a_corrected) < error_iterations and np.abs(a_line - a_line_corrected) < error_iterations:
            break
    c_t = 4 * a_corrected * (1 - a_corrected)
    c_q = 4 * a_line_corrected * (1 - a_corrected) * r_over_R * tip_speed_ratio
    c_p = c_t * (1 - a)

    if yaw_angle != 0:
        dpsi = psi_vec[1:] - psi_vec[:-1]
        dpsi = np.append(dpsi, dpsi[-1])
        c_t = np.zeros(len(psi_vec))
        c_q = np.zeros(len(psi_vec))
        normal_force = np.zeros(len(psi_vec))
        tangential_force = np.zeros(len(psi_vec))
        gamma = np.zeros(len(psi_vec))
        a_corrected = np.zeros(len(psi_vec))
        a_line_corrected = np.zeros(len(psi_vec))
        alpha = np.zeros(len(psi_vec))
        inflow_angle = np.zeros(len(psi_vec))

        for i, psi in enumerate(psi_vec):
            wake_skew_angle = (0.6*a_new + 1) * yaw_angle
            a_skew = 2 * math.tan(wake_skew_angle/2) * r_over_R * math.sin(psi)
            a_new = a_new + a_skew
            a_line_skew = 1/r_over_R * 1/tip_radius_over_R * math.sin(psi) * math.sin(yaw_angle)
            a_line_new = a_line_new + a_line_skew

            # correct new axial induction with Prandtl's correction
            prandtl, prandtl_tip, prandtl_root = prandtl_tip_root_correction(
                r_over_R, root_radius_over_R, tip_radius_over_R, omega * rotor_R / u_inf, blades_number, a_new)
            if prandtl < 0.0001: prandtl = 0.0001   # avoid divide by zero
            a_corrected[i] = a_new / prandtl                     # correct estimate of axial induction
            a_line_corrected[i] = a_line_new / prandtl           # correct estimate of azimuthal induction
            c_t = 4 * a_corrected[i] * (1 - a_corrected[i]*(2*math.cos(yaw_angle - a)))
            c_p = c_t * (math.cos(yaw_angle) - a)
            c_q = 4 * a_line_corrected[i] * (1 - a) * r_over_R * tip_speed_ratio * (math.cos(psi)**2 + (math.cos(wake_skew_angle)**2) * (math.sin(psi))**2 )
            normal_force[i] = c_t * 0.5 * u_inf**2 * np.pi * rotor_R**2 * dr * dpsi[i] * r_over_R * rotor_R
            tangential_force[i] = c_q * 0.5 * u_inf**2 * np.pi * rotor_R**2 * dr * dpsi[i] * r_over_R * rotor_R
            inflow_angle[i] = np.arctan2(u_inf*(math.cos(yaw_angle) - a_corrected[i]), omega*r_over_R*rotor_R*(1 + a_line_corrected[i]) + u_inf*math.sin(yaw_angle)*math.sin(psi))
            alpha[i] = inflow_angle[i] * 180 / np.pi - twist

    results = {
        'a': a_corrected, 
        'a_line': a_line_corrected, 
        'r_over_R': r_over_R,
        'normal_force': normal_force, 
        'tangential_force': tangential_force, 
        'gamma': gamma,
        'alpha': alpha,
        'inflow_angle': inflow_angle,
        'c_thrust': c_t,
        'c_torque': c_q,
        'c_power': c_p,
        'psi': 0 if yaw_angle == 0 else psi_vec,
        'c_thrust_iterations': c_t_iterations[~np.isnan(c_t_iterations)]
        }
    
    return results