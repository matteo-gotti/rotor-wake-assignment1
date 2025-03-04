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

def compute_c_t_yawed(a, yaw_angle, wake_skew_angle):
    """
    This function calculates the thrust coefficient as a function of induction factor 'a', according to vortex theory
    """
    c_t = 4*a*(math.cos(yaw_angle) + math.tan(wake_skew_angle/2)*math.sin(yaw_angle) - a*sec(wake_skew_angle/2)**2)
    return c_t

def compute_c_p_yawed(a, yaw_angle, wake_skew_angle):
    """
    This function calculates the power coefficient as a function of induction factor 'a', according to vortex theory
    """
    c_p = 4*a*(math.cos(yaw_angle) + math.tan(wake_skew_angle/2)*math.sin(yaw_angle) - a*sec(wake_skew_angle/2)**2)*(math.cos(yaw_angle) - a)
    return c_p

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
    given a root and tip radius (also non-dimensioned), a tip speed ratio TSR, the number lf blades NBlades and the axial induction factor
    """
    temp1 = -blades_number / 2 * (tip_radius_over_R - r_R) / r_R * np.sqrt(1 + ((tip_speed_ratio * r_R) ** 2) / ((1 - axial_induction) ** 2))
    f_tip = np.array(2 / np.pi * np.arccos(np.exp(temp1)))
    f_tip[np.isnan(f_tip)] = 0
    temp1 = blades_number / 2 * (root_radius_over_R - r_R) / r_R * np.sqrt(1 + ((tip_speed_ratio * r_R) ** 2) / ((1 - axial_induction) ** 2))
    f_root = np.array(2 / np.pi * np.arccos(np.exp(temp1)))
    f_root[np.isnan(f_root)] = 0
    return f_root * f_tip, f_tip, f_root

def load_blade_element(v_normal, v_tangential, chord, twist, polar_alpha, polar_cl, polar_cd):
    """
    calculates the load in the blade element
    """
    v_magnitude_squared = v_normal ** 2 + v_tangential ** 2
    inflow_angle = np.arctan2(v_normal, v_tangential)
    alpha = twist + inflow_angle * 180 / np.pi
    cl = np.interp(alpha, polar_alpha, polar_cl)
    cd = np.interp(alpha, polar_alpha, polar_cd)
    lift = 0.5 * v_magnitude_squared * cl * chord
    drag = 0.5 * v_magnitude_squared * cd * chord
    normal_force = lift * np.cos(inflow_angle) + drag * np.sin(inflow_angle)
    tangential_force = lift * np.sin(inflow_angle) - drag * np.cos(inflow_angle)
    gamma = 0.5 * np.sqrt(v_magnitude_squared) * cl * chord
    return normal_force, tangential_force, gamma

def solve_stream_tube(u_inf, r1_over_R, r2_over_R, root_radius_over_R, tip_radius_over_R, omega, rotor_R, blades_number,
                      chord, twist, polar_alpha, polar_cl, polar_cd):
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
    area = np.pi * ((r2_over_R * rotor_R) ** 2 - (r1_over_R * rotor_R) ** 2)  # area stream-tube
    r_over_R = (r1_over_R + r2_over_R) / 2  # centroid

    # initialize variables
    normal_force = np.nan
    tangential_force = np.nan
    gamma = np.nan
    a = 0.0  # axial induction
    a_line = 0.0  # tangential induction factor
    n_iterations = 100
    error_iterations = 0.00001  # error limit for iteration process, in absolute value of induction
    for i in range(n_iterations):
        # ///////////////////////////////////////////////////////////////////////
        # // this is the block "Calculate velocity and loads at blade element"
        # ///////////////////////////////////////////////////////////////////////
        u_rotor = u_inf * (1 - a)  # axial velocity at rotor
        u_tangential = (1 + a_line) * omega * r_over_R * rotor_R  # tangential velocity at rotor
        # calculate loads in blade segment in 2D (N/m)
        normal_force, tangential_force, gamma = load_blade_element(u_rotor, u_tangential, chord, twist, polar_alpha, polar_cl, polar_cd)
        load_3d_axial = normal_force * rotor_R * (r2_over_R - r1_over_R) * blades_number  # 3D force in axial direction
        # load_3d_tangential =loads[1]*Radius*(r2_R-r1_R)*NBlades # 3D force in azimuthal/tangential direction (not used here)

        # ///////////////////////////////////////////////////////////////////////
        # //the block "Calculate velocity and loads at blade element" is done
        # ///////////////////////////////////////////////////////////////////////

        # ///////////////////////////////////////////////////////////////////////
        # // this is the block "Calculate new estimate of axial and azimuthal induction"
        # ///////////////////////////////////////////////////////////////////////
        # // calculate thrust coefficient at the stream-tube
        c_t = load_3d_axial / (0.5 * area * u_inf ** 2)

        # calculate new axial induction, accounting for Glauert's correction
        a_new = compute_axial_induction(c_t)

        # correct new axial induction with Prandtl's correction
        prandtl, prandtl_tip, prandtl_root = prandtl_tip_root_correction(r_over_R, root_radius_over_R, tip_radius_over_R,
                                                                       omega * rotor_R / u_inf, blades_number, a_new)
        if prandtl < 0.0001:prandtl = 0.0001  # avoid divide by zero
        a_new = a_new / prandtl  # correct estimate of axial induction
        a = 0.75 * a + 0.25 * a_new  # for improving convergence, weigh current and previous iteration of axial induction

        # calculate azimuthal induction
        a_line = tangential_force * blades_number / (2 * np.pi * u_inf * (1 - a) * omega * 2 * (r_over_R * rotor_R) ** 2)
        a_line =  a_line / prandtl  # correct estimate of azimuthal induction with Prandtl's correction
        # ///////////////////////////////////////////////////////////////////////////
        # // end of the block "Calculate new estimate of axial and azimuthal induction"
        # ///////////////////////////////////////////////////////////////////////

        # // test convergence of solution, by checking convergence of axial induction
        if np.abs(a - a_new) < error_iterations:
            # print("iterations")
            # print(i)
            break

    return [a, a_line, r_over_R, normal_force, tangential_force, gamma]