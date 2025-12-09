import numpy as np

def beam_range_finder_model(Xt, z, map_data, compute_from_map,
                            PDF=np.random.normal,
                             min_theta = -np.pi/2, max_theta = np.pi/2 ,
                            max_range=10.0, sigma_hit=0.2, lambda_short=0.1, 
                            z_hit=0.7, z_short=0.1, z_max=0.1, z_rand=0.1):
    x, y, theta = Xt
    prob = 1.0
    
    for i in range(len(z)):

        angle = theta + min_theta + i * (max_theta - min_theta) / (len(z) - 1)
        z_expected = compute_from_map(x, y, angle, map_data, max_range)
        z_measured = z[i]

        # Hit component
        p_hit = PDF(z_measured - z_expected, sigma_hit) if 0 <= z_measured <= max_range else 0

        # Short component
        p_short = lambda_short * np.exp(-lambda_short * z_measured) if 0 <= z_measured <= z_expected else 0

        # Max range component
        p_max = 1.0 if z_measured == max_range else 0

        # Random component
        p_rand = 1.0 / max_range if 0 <= z_measured < max_range else 0

        p = z_hit * p_hit + z_short * p_short + z_max * p_max + z_rand * p_rand
        prob *= p

    return prob

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

def learn_intrinsic_params(sensor_data, true_positions, initial_params, sensor_model, learning_rate=0.01, num_iterations=100):

    params = initial_params.copy()

    for _ in range(num_iterations):
        gradients = {key: 0.0 for key in params.keys()}

        for z, pos in zip(sensor_data, true_positions):
            z_expected = sensor_model(pos, params)
            error = z - z_expected

            for key in params.keys():
                # Numerical gradient approximation
                delta = 1e-5
                params_plus = params.copy()
                params_plus[key] += delta
                z_expected_plus = sensor_model(pos, params_plus)
                error_plus = z - z_expected_plus

                gradients[key] += (error_plus - error) / delta

        # Update parameters
        for key in params.keys():
            params[key] -= learning_rate * gradients[key] / len(sensor_data)
    return params

