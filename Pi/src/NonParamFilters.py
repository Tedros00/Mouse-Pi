import numpy as np

def discrete_bayes_filter(prev_belief_vec, u, z, motion_model, sensor_model):

    belief_bar = np.zeros_like(prev_belief_vec)
    belief_vec = np.zeros_like(prev_belief_vec)
    
    # Prediction step
    for k in range(len(prev_belief_vec)):
        belief_bar[k] = motion_model(prev_belief_vec, u, k)
    
    # Update step
    for k in range(len(prev_belief_vec)):
        belief_vec[k] = sensor_model(belief_bar, z, k)
    
    # Normalize
    belief_vec /= np.sum(belief_vec) + 1e-10  # Add epsilon to avoid division by zero
    return belief_vec

def particle_filter(Prev_particles, u, z, motion_model, sensor_model, num_particles):

    particles = np.zeros_like(Prev_particles)
    weights = np.zeros(num_particles)
    
    # Prediction step
    for i in range(num_particles):
        particles[:, i:i+1] = motion_model(Prev_particles[:, i:i+1], u)
    
    # Update step - compute weights
    for i in range(num_particles):
        weights[i] = sensor_model(particles[:, i:i+1], z)
    
    # Normalize weights
    weights /= np.sum(weights) + 1e-10
    
    # Low variance resampling step
    new_particles = np.zeros_like(particles)
    r = np.random.uniform(0, 1.0 / num_particles)
    c = weights[0]
    j = 0
    
    for i in range(num_particles):
        u_i = r + i / num_particles
        while u_i > c and j < num_particles - 1:
            j += 1
            c += weights[j]
        new_particles[:, i:i+1] = particles[:, j:j+1]
    
    particles = new_particles
    return particles
