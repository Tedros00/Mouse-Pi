import numpy as np

def velocity_motion_model(Xt, ut, prev_Xt, PDF=np.random.normal, dt = 1e-3, alphas=(0.01, 0.01, 0.01, 0.01) ):

    x0,y0,theta0 = prev_Xt
    x1,y1,theta1 = Xt
    v, w = ut

    #compute Mu
    Mu = 1/2 * ((x0-x1)*np.cos(theta0) + (y0-y1)*np.sin(theta0))/((y0-y1)*np.cos(theta0) - (x0-x1)*np.sin(theta0))
    
    #center of rotation
    xc = (x0+x1)/2 + Mu * (y0-y1)
    yc = (y0+y1)/2 - Mu * (x0-x1)
    rc = np.hypot(x0 - xc, y0 - yc)

    delta_theta = np.arctan2(y1 - yc, x1 - xc) - np.arctan2(y0 - yc, x0 - xc)

    #compute expected control
    v_hat = rc * delta_theta/dt
    omega_hat = delta_theta/dt
    gamma_hat = (theta1-theta0)/dt  - omega_hat

    #compute probability
    alpha1, alpha2, alpha3, alpha4 = alphas

    probv     = PDF(v - v_hat, alpha1 * abs(v_hat) + alpha2 * abs(omega_hat)) 
    probw     = PDF(w - omega_hat, alpha3 * abs(v_hat) + alpha4 * abs(omega_hat))
    probgamma = PDF(gamma_hat, alpha1 * abs(v_hat) + alpha2 * abs(omega_hat))

    prob = probv * probw * probgamma
    return prob

def sample_motion_velocity_model(prev_Xt, ut, PDF=np.random.normal, dt = 1e-3, alphas=(0.01, 0.01, 0.01, 0.01, 0.01, 0.01) ):

    x0,y0,theta0 = prev_Xt
    v, w = ut

    #add noise to control
    alpha1, alpha2, alpha3, alpha4, alpha5, alpha6 = alphas
    v_hat = v + PDF(0, alpha1*abs(v) + alpha2*abs(w))
    w_hat = w + PDF(0, alpha3*abs(v) + alpha4*abs(w))
    gamma_hat = PDF(0, alpha5*abs(v) + alpha6*abs(w))

    #compute new state
    if abs(w_hat) > 1e-6:
        x1 = x0 - (v_hat/w_hat) * np.sin(theta0) + (v_hat/w_hat) * np.sin(theta0 + w_hat*dt)
        y1 = y0 + (v_hat/w_hat) * np.cos(theta0) - (v_hat/w_hat) * np.cos(theta0 + w_hat*dt)
    else:
        x1 = x0 + v_hat * dt * np.cos(theta0)
        y1 = y0 + v_hat * dt * np.sin(theta0)
    theta1 = theta0 + w_hat * dt + gamma_hat * dt

    Xt = np.array([x1, y1, theta1])
    return Xt

def odometry_motion_model(Xt, ut, prev_Xt, PDF=np.random.normal, alphas=(0.01, 0.01, 0.01) ):

    x0,y0,theta0 = prev_Xt
    x1,y1,theta1 = Xt
    delta_rot1, delta_trans, delta_rot2 = ut

    #compute expected control
    delta_rot1_hat = np.arctan2(y1 - y0, x1 - x0) - theta0
    delta_trans_hat = np.hypot(x1 - x0, y1 - y0)
    delta_rot2_hat = theta1 - theta0 - delta_rot1_hat 

    #compute probability
    alpha1, alpha2, alpha3 = alphas 
    prob_rot1   = PDF(delta_rot1 - delta_rot1_hat, alpha1 * abs(delta_rot1_hat) + alpha2 * abs(delta_trans_hat))    
    prob_trans  = PDF(delta_trans - delta_trans_hat, alpha3 * abs(delta_trans_hat) + alpha2 * (abs(delta_rot1_hat) + abs(delta_rot2_hat)))
    prob_rot2   = PDF(delta_rot2 - delta_rot2_hat, alpha1 * abs(delta_rot2_hat) + alpha2 * abs(delta_trans_hat))    
    prob = prob_rot1 *  prob_trans * prob_rot2
    return prob 

def sample_odometry_motion_model(prev_Xt, ut, dt = 1e-3, PDF=np.random.normal, alphas=(0.01, 0.01, 0.01) ):

    x0,y0,theta0 = prev_Xt
    delta_rot1, delta_trans, delta_rot2 = ut
    
    #add noise to control
    alpha1, alpha2, alpha3 = alphas
    delta_rot1_hat = delta_rot1 + PDF(0, alpha1 * abs(delta_rot1) + alpha2 * abs(delta_trans))
    delta_trans_hat = delta_trans + PDF(0, alpha3 * abs(delta_trans)) 
    delta_rot2_hat = delta_rot2 + PDF(0, alpha1 * abs(delta_rot2) + alpha2 * abs(delta_trans))

    #compute new state
    x1 = x0 + delta_trans_hat * np.cos(theta0 + delta_rot1_hat)
    y1 = y0 + delta_trans_hat * np.sin(theta0 + delta_rot1_hat)
    theta1 = theta0 + delta_rot1_hat + delta_rot2_hat   
    Xt = np.array([x1, y1, theta1])
    return Xt

def command_correction(
        u_cmd,     # (v_c, w_c)     commanded control
        u_fb,      # (v_m, w_m)     encoder feedback
        sigma_cmd = (0.0039, 0.0572), # (σ_c_v, σ_c_w) command noise std
        sigma_fb = (0.0010, 0.0115),  # (σ_m_v, σ_m_w) encoder noise std
        cmd_threshold = 0.001  # Deadband threshold
        #run calibrate_sigmas.py to get these values
    ):
    """
    Computes the posterior distribution p(u | u_cmd, u_fb)
    where u = (v, w).
    
    All variables are treated as Gaussians.
    When command is near zero (below threshold), return command without fusion
    to avoid drift from encoder noise.
    
    Returns:
        u_fused_mean  = (v_fused, w_fused)
        u_fused_cov   = 2x2 covariance matrix
    """

    # unpack
    v_c, w_c = u_cmd
    v_m, w_m = u_fb
    sigma_c_v, sigma_c_w = sigma_cmd
    sigma_m_v, sigma_m_w = sigma_fb

    # --- compute fusion weights (Kalman-style) ---
    lam_v = sigma_c_v**2 / (sigma_c_v**2 + sigma_m_v**2)
    lam_w = sigma_c_w**2 / (sigma_c_w**2 + sigma_m_w**2)

    # --- fused mean with deadband per component ---
    # Only fuse v if command is significant, else use command
    if abs(v_c) < cmd_threshold:
        v_fused = v_c
    else:
        v_fused = (1 - lam_v) * v_c + lam_v * v_m
    
    # For angular velocity, only fuse if feedback is reasonably large
    # This prevents the fusion from dampening small commanded rotations
    if abs(w_c) < cmd_threshold or abs(w_m) < cmd_threshold * 1000:
        w_fused = w_c
    else:
        w_fused = (1 - lam_w) * w_c + lam_w * w_m

    return np.array([v_fused, w_fused])
