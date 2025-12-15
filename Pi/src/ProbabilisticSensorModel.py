import numpy as np
from scipy import stats

def scan_likelihood(z_measured, z_expected, sigma_hit=0.05):
    """
    Compute likelihood of a measurement scan given an expected scan.
    
    This uses the SAME approach as the plotting code - direct full-scan comparison.
    This is the correct way to implement the beam range finder model.
    
    Parameters:
    - z_measured: actual measurement (array of 360 ranges in meters)
    - z_expected: expected measurement from particle pose (array of 360 ranges)
    - sigma_hit: standard deviation of Gaussian (meters)
    
    Returns:
    - likelihood in [0, 1]
    """
    total_prob = 0.0
    
    for i in range(len(z_measured)):
        # Gaussian: exp(-(z_meas - z_exp)^2 / (2*sigma^2))
        # This measures how well the expected scan matches the measured scan
        gaussian = np.exp(-((z_measured[i] - z_expected[i]) ** 2) / (2 * sigma_hit ** 2))
        total_prob += gaussian
    
    # Average across all beams to get final likelihood
    avg_prob = total_prob / len(z_measured) if len(z_measured) > 0 else 0
    return np.clip(avg_prob, 0, 1)


def beam_range_finder_model(Xt, z, map_data, compute_from_map,
                            PDF=np.random.normal,
                             min_theta = -np.pi, max_theta = np.pi ,
                            max_range=10.0, sigma_hit=0.05, lambda_short=0.01, 
                            z_hit=0.95, z_short=0.05, z_max=0.01, z_rand=0.01):
    """
    DEPRECATED: Use scan_likelihood(z_measured, z_expected) instead.
    
    The proper approach is to:
    1. Compute z_gt = compute_expected_measurements(gt_pose, ...) once
    2. For each particle: z_expected = compute_expected_measurements(particle_pose, ...)
    3. Call scan_likelihood(z_gt, z_expected)
    
    This ensures both use the EXACT SAME raycasting function.
    """
    x, y, theta = Xt
    total_prob = 0.0
    
    for i in range(len(z)):
        # Beam angle relative to robot heading
        beam_angle = theta + (i - len(z)/2) * (2 * np.pi / len(z))
        
        # Expected measurement for this beam from the map
        z_expected = compute_from_map(x, y, beam_angle, map_data, max_range)
        z_measured = z[i]
        
        # Hit component: unnormalized Gaussian
        if 0 < z_measured < max_range:
            # Gaussian: exp(-(z - mu)^2 / (2*sigma^2))
            gaussian = np.exp(-((z_measured - z_expected) ** 2) / (2 * sigma_hit ** 2))
            p_hit = gaussian
        else:
            p_hit = 0
        
        # Accumulate probability
        total_prob += z_hit * p_hit

    # Average probability across all beams
    avg_prob = total_prob / (len(z) * z_hit) if len(z) > 0 else 0
    
    return np.clip(avg_prob, 0, 1)

def likelyhood_field_model(Xt,
                            z,
                            get_distance_to_closest_obstacle ,
                            map_data, sigma=0.5, z_max=10.0):
    x, y, theta = Xt
    prob = 1.0

    for i in range(len(z)):
        angle = theta + (i - len(z)/2) * (np.pi / len(z))  # Assuming uniform angular distribution
        z_measured = z[i]
        if z_measured < z_max:
            x_z = x + z_measured * np.cos(angle)
            y_z = y + z_measured * np.sin(angle)
            d = get_distance_to_closest_obstacle(x_z, y_z, map_data)
            p = np.exp(- (d ** 2) / (2 * sigma ** 2))
            prob *= p

    return prob