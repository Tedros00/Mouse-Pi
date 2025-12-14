import numpy as np

def grid_localization(prev_belief, u, z, motion_model, sensor_model):

    belief_bar = np.zeros_like(prev_belief)
    belief = np.zeros_like(prev_belief)
    
    # Prediction step: accumulate motion model contributions
    for k in range(len(prev_belief)):
        belief_bar[k] = motion_model(prev_belief, u, k)
    
    # Update step: apply sensor model
    for k in range(len(prev_belief)):
        belief[k] = sensor_model(belief_bar, z, k)
    
    # Normalize
    belief_sum = np.sum(belief)
    if belief_sum > 0:
        belief /= belief_sum
    else:
        belief = np.ones_like(belief) / len(belief)
    
    return belief

def monte_carlo_localization(prev_Xi, u, z, sample_motion_model, sensor_model, m, compute_from_map=None):
    """
    Monte Carlo Localization with low-variance resampling.
    
    Args:
        prev_Xi: Previous particles (N x 3) array with [x, y, theta]
        u: Control input [v, w]
        z: Sensor measurement (LiDAR ranges)
        sample_motion_model: Function to sample from motion model
        sensor_model: Function to compute particle weight from sensor data
        m: Map data
        compute_from_map: Optional function (not used in this implementation)
    
    Returns:
        Xi_resampled: Resampled particles
        weights: Normalized particle weights
    """
    num_particles = len(prev_Xi)
    Xi = np.zeros_like(prev_Xi)
    weights = np.zeros(num_particles)

    # Motion model: update all particles with control input
    for i in range(num_particles):
        Xi[i] = sample_motion_model(prev_Xi[i], u, dt=0.1)
    
    # Sensor model: weight particles based on measurement likelihood
    for i in range(num_particles):
        pose = Xi[i]
        weights[i] = sensor_model(pose, z, m, sigma_hit=0.1)

    # Normalize weights
    weights_sum = np.sum(weights)
    if weights_sum > 0:
        weights /= weights_sum
    else:
        weights = np.ones(num_particles) / num_particles

    # Low-variance resampling
    cumsum = np.cumsum(weights)
    r = np.random.uniform(0, 1.0 / num_particles)
    j = 0
    Xi_resampled = np.zeros_like(Xi)
    for i in range(num_particles):
        u_i = r + i / num_particles
        while j < num_particles - 1 and u_i > cumsum[j]:
            j += 1
        Xi_resampled[i] = Xi[j]

    return Xi_resampled, weights

def augmented_monte_carlo_localization(prev_Xi, u, z, sample_motion_model, sensor_model, m, compute_from_map, alpha_slow=0.001, alpha_fast=0.1):

    num_particles = len(prev_Xi)
    Xi = np.zeros_like(prev_Xi)
    Xi_bar = np.zeros_like(prev_Xi)
    w_slow = 0.0
    w_fast = 0.0
    wavg = 0.0

    for i in range(num_particles):

        #sample from motion model
        X = sample_motion_model(prev_Xi[i], u, PDF=np.random.normal, dt = 1e-3, alphas=(0.1, 0.1, 0.1, 0.1) )
        #new weights from sensor model
        w = sensor_model (X, z, m, compute_from_map,
                            PDF=np.random.normal,
                            min_theta = -np.pi/2, max_theta = np.pi/2 ,
                            max_range=10.0, sigma_hit=0.2, lambda_short=0.1, 
                            z_hit=0.7, z_short=0.1, z_max=0.1, z_rand=0.1)
        #add to Xi
        Xi_bar[i]= (X, w)

        #update wavg
        wavg+=1/num_particles * w

    #update w_slow and w_fast
    w_slow += alpha_slow*(wavg - w_slow)
    w_fast += alpha_fast*(wavg - w_fast)

    # Resample particles based on weights
    for m in range(num_particles):
        weights = np.array([Xi_bar[i][1] for i in range(num_particles)])
        weights /= np.sum(weights)  # Normalize weights
        prob_random = max(0.0, 1.0 - w_fast / w_slow) if w_slow > 0 else 1.0
        if np.random.rand() < prob_random:
            # Sample a random particle from the entire state space
            Xi[m] = np.array([np.random.normal(-5, 5), np.random.normal(-5, 5), np.random.normal(-np.pi, np.pi)])  # Example bounds
        else:
            index = np.random.choice(range(num_particles), p=weights)
            Xi[m] = Xi_bar[index][0]

    return Xi, weights, w_slow, w_fast

def KLD_sampling_monte_carlo_localization(prev_Xi, u, z, sample_motion_model, sensor_model, m, compute_from_map, epsilon=0.05, delta=0.99):

    num_particles = len(prev_Xi)
    Xi = np.zeros_like(prev_Xi)
    Xi_bar = np.zeros_like(prev_Xi)

    for i in range(num_particles):

        Xi = np.zeros_like(prev_Xi)
        M = 0
        M_Xi = 0
        H = np.zeros_like(prev_Xi)
        M_min = 0

        while(M < M_Xi or M < M_min):

            #sample from motion model
            X = sample_motion_model(prev_Xi[i], u, PDF=np.random.normal, dt = 1e-3, alphas=(0.1, 0.1, 0.1, 0.1) )
            #new weights from sensor model
            w = sensor_model (X, z, m, compute_from_map,
                                PDF=np.random.normal,
                                min_theta = -np.pi/2, max_theta = np.pi/2 ,
                                max_range=10.0, sigma_hit=0.2, lambda_short=0.1, 
                                z_hit=0.7, z_short=0.1, z_max=0.1, z_rand=0.1)
            #add to Xi
            Xi_bar[i]= (X, w)

            if X not in H:
                k += 1
                H.append(X)
                if k > 1:
                    M_Xi = (k-1/epsilon)*(1 - 2/(9*(k-1)) + np.sqrt(2/(9*(k-1)))*np.random.normal())**3

            M += 1
    return Xi