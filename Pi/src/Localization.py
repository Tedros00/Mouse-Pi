import numpy as np    

def markov_localization(prev_belief, u, z, motion_model, sensor_model):

    belief_bar = np.zeros_like(prev_belief)
    belief = np.zeros_like(prev_belief)
    
    
    for k in range(len(prev_belief)):
        # Prediction step
        belief_bar += motion_model(prev_belief, u, k)
        # Update step
        belief += sensor_model(belief_bar, z, k)
    
    # Normalize
    belief /= np.sum(belief)
    return belief

def ekf_localization(prev_Mu, Prev_Sigma, u, z, m, sample_velocity_model, sample_sensor_model, G_jacobian, u_jacobian , z_Jacobian, Q, alphas = (0.1, 0.1, 0.1, 0.1), sensor_model= None, compute_from_map= None):

    alpha1, alpha2, alpha3, alpha4 = alphas
    # Prediction step
    G = G_jacobian(prev_Mu, u)

    #covariance of motion noise
    M = np.array([[alpha1 * u[0]**2 + alpha2 * u[1]**2, 0],
                  [0, alpha3 * u[0]**2 + alpha4 * u[1]**2]])
    
    #V = Jacobian of motion control
    V = u_jacobian(prev_Mu, u)

    #predicted pose after motion
    Mu_bar = sample_velocity_model(prev_Mu, u, PDF=np.random.normal, dt = 1e-3 , alphas=alphas)
    
    #compute predicted covariance
    Sigma_bar = G @ Prev_Sigma @ G.T + V @ M @ V.T

    #Q is passed by the function parameter
    
    for i in range(len(z)):
        #assign j to the corresponding landmark
        j = m['id_to_landmark'][i]
        q = (Mu_bar[0] - m['landmarks'][j][0])**2 + (Mu_bar[1] - m['landmarks'][j][1])**2

        #predict measurement
        z_pred = sample_sensor_model(Mu_bar, m, i)

        #measurement Jacobian
        H = z_Jacobian(Mu_bar, m, i)

        #measurement prediction
        z_pred = sample_sensor_model(Mu_bar, m, i)

        #uncertainty in measurement
        S = H @ Sigma_bar @ H.T + Q
        #Kalman Gain
        K = Sigma_bar @ H.T @ np.linalg.inv(S)

        #update pose with measurement
        Mu_bar = Mu_bar + K @ (z[i] - z_pred)

        #update covariance
        Sigma_bar = (np.eye(len(K)) - K @ H) @ Sigma_bar
    
    Mu = Mu_bar
    Sigma = Sigma_bar

    #compute final likelyhood of measurement
    prob = sensor_model(Mu, z, m, compute_from_map ,PDF=np.random.normal,min_theta = -np.pi/2, max_theta = np.pi/2 , max_range=10.0, sigma_hit=0.2, lambda_short=0.1, z_hit=0.7, z_short=0.1, z_max=0.1, z_rand=0.1)

    return Mu, Sigma, prob
