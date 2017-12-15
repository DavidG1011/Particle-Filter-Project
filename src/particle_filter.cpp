/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;


void ParticleFilter::init(double x, double y, double theta, double std[])
{

	// Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).


    // Init particle vector
    Particle pInit;


    // Potentially bad estimate: modify if needed.
    num_particles = 100;

    // Init random gen
    default_random_engine gen;

    // Create gaussian dists.
    normal_distribution<double> normdist_x(x, std[0]);
    normal_distribution<double> normdist_y(y, std[1]);
    normal_distribution<double> normdist_theta(theta, std[2]);

    for (int n = 0; n < num_particles; n++)
    {
        // Add particles with gaussian noise
        pInit.id = n;
        pInit.x = normdist_x(gen);
        pInit.y = normdist_y(gen);
        pInit.theta = normdist_theta(gen);
        pInit.weight = 1;

        particles.push_back(pInit);
    }

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate)
{
	// Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

    // Init random gen.
    default_random_engine gen;

    // Create gaussian dists.
    normal_distribution<double> normdist_x(0, std_pos[0]);
    normal_distribution<double> normdist_y(0, std_pos[1]);
    normal_distribution<double> normdist_theta(0, std_pos[2]);

    for (int n = 0; n < num_particles; n++)
    {
        // Prevent division by zero if yaw rate = 0.
        if (abs(yaw_rate) != 0)
        {
            // Update x positions.
            particles[n].x += velocity / yaw_rate * (sin(particles[n].theta + yaw_rate*(delta_t)) -
                              sin(particles[n].theta));

            // Update y positions.
            particles[n].y += velocity / yaw_rate * (cos(particles[n].theta) - cos(particles[n].theta +
                              yaw_rate * delta_t));
            // Update theta
            particles[n].theta += yaw_rate * delta_t;

        }
        else
        {
            particles[n].x += velocity * delta_t * cos(particles[n].theta);

            particles[n].y += velocity * delta_t * sin(particles[n].theta);
        }

        // Add gaussian noise.
        particles[n].x += normdist_x(gen);
        particles[n].y += normdist_y(gen);
        particles[n].theta += normdist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations)
{
	// Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

    // Loop observations
    for (unsigned int n = 0; n < observations.size(); n++)
    {
        // Init min and closest ID to be non-determinate
        int closestid = -1;
        double min = 99999999;
        LandmarkObs obs = observations[n];

        // Loop predictions
        for (unsigned int n2 = 0; n2 < predicted.size(); n2++)
        {
            // Measure dist between observation and prediction positions
            LandmarkObs pred = predicted[n2];
            double distance = dist(obs.x, obs.y, pred.x, pred.y);

            // Find closest prediction point and assign it's ID
            if (distance < min)
            {
                min = distance;
                closestid = pred.id;
            }

        }
        // Assign observation ID to closest predicted ID
        observations[n].id = closestid;
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const std::vector<LandmarkObs> &observations, const Map &map_landmarks)
{
	// Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

    for (int n = 0; n < num_particles; n++)
    {
        vector<LandmarkObs> inrangelm;

        // Unpack particle positions and IDs for readability
        double px = particles[n].x;
        double py = particles[n].y;
        double ptheta = particles[n].theta;

        // Loop map landmarks
        for (unsigned int n2 = 0; n2 < map_landmarks.landmark_list.size(); n2++)
        {
            // Unpack landmark positions and ID for readability
            float m_x = map_landmarks.landmark_list[n2].x_f;
            float m_y = map_landmarks.landmark_list[n2].y_f;
            int m_id = map_landmarks.landmark_list[n2].id_i;

            // Check if distance from particle to landmark is in sensor range
//            if (abs(m_x - px) <= sensor_range && abs(m_y - py) <= sensor_range)
//                if (dist(px,py,m_x,m_y) <= sensor_range)
//            {
//                // Push back landmark position if in range of particle.
//                inrangelm.push_back(LandmarkObs{m_id, m_x, m_y});
//            }
            if (dist(m_x,m_y,px,py) <= sensor_range)
            {
                inrangelm.push_back(LandmarkObs{m_id, m_x, m_y});
            }
        }

        // Create vector for holding points remapped from the car coord system to the map coord system.
        vector<LandmarkObs> remapped_obs;

        for (unsigned int n2 = 0; n2 < observations.size(); n2++)
        {
            // Remap observations from vehicle coords to map coords
            double r_x = px + (cos(ptheta)*observations[n2].x) - (sin(ptheta)*observations[n2].y);
            double r_y = py + (sin(ptheta)*observations[n2].x) + (cos(ptheta)*observations[n2].y);

            // Push back remapped points and observation ID (ID unchanged)
            remapped_obs.push_back(LandmarkObs{observations[n2].id, r_x, r_y});
        }

        // Find remapped observation that is closest to predicted landmark
        dataAssociation(inrangelm, remapped_obs);


        // Reset weight to 1
        particles[n].weight = 1;

        for (unsigned int n2 = 0; n2 < remapped_obs.size(); n2++)
        {
            // Unpack coords for observation points
            double x = remapped_obs[n2].x;
            double y = remapped_obs[n2].y;

            // Init vars to hold coords of nearest landmarks
            double mu_x;
            double mu_y;

            // Get ID of remapped observation point
            int associated = remapped_obs[n2].id;

            // Loop through predicted landmarks
            for (unsigned int n3 = 0; n3 < inrangelm.size(); n3++)
            {
                // Get coords of prediction closest to observed landmark to "assign" the guess
                if (inrangelm[n3].id == associated)
                {
                    // Unpack mu coords
                    mu_x = inrangelm[n3].x;
                    mu_y = inrangelm[n3].y;
                }
            }

            // Unpack landmark uncertainties
            double sig_x = std_landmark[0];
            double sig_y = std_landmark[1];

            // Calculate weight using multi-variate Gaussian distribution
            double gauss_norm = (1/(2 * M_PI * sig_x * sig_y));
            double exponent = exp(-(pow(x - mu_x, 2) / (2 * pow(sig_x, 2)) +
                                 (pow(y - mu_y, 2) / (2 * pow(sig_y, 2)))));

            double w = gauss_norm * exponent;


            // Accumulate total weight product
            particles[n].weight *= w;
        }
    }
}

void ParticleFilter::resample()
{
	// Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    // Create vector of current particle weights.
    vector<double> weights;

    // Create new particle vector for resampled particles
    vector<Particle> resampledP;

    for (int n = 0; n < num_particles; n++)
    {
        weights.push_back(particles[n].weight);
    }

    // Create random num gen
    default_random_engine gen;

    // Create discrete distribution to produce values with a predefined probability from their weight
    discrete_distribution<> weightdist(weights.begin(),weights.end());

    // Loop through each particle
    for (int n = 0; n < num_particles; ++n)
    {
        // Push back particles resampled randomly from discrete distribution
        resampledP.push_back(particles[weightdist(gen)]);
    }

    // Assign resampled particles to particle list
    particles = resampledP;
}


Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
