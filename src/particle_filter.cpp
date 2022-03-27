/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

#define EPSILON (0.00001)

// Create a random engine to take samples from normal distributions
static std::default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */

  if (is_initialized != true) {
  
    num_particles = 50;

    // Create normal distributions for x, y and theta
    std::normal_distribution<double> dist_x(x, std[0]);
    std::normal_distribution<double> dist_y(y, std[1]);
    std::normal_distribution<double> dist_theta(theta, std[2]);


    for (unsigned int i = 0; i < num_particles; i++) {

      // Create a particle with samples from normal distributions:
      Particle particle;
      particle.id = i;
      particle.x = dist_x(gen);
      particle.y = dist_y(gen);
      particle.theta = dist_theta(gen);
      particle.weight = 1.0;

      // Add particle to list
      particles.push_back(particle);
    }

    this->is_initialized = true;
  }
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */

  // Create zero mean normal distributions for x, y and theta
  std::normal_distribution<double> dist_x(0, std_pos[0]);
  std::normal_distribution<double> dist_y(0, std_pos[1]);
  std::normal_distribution<double> dist_theta(0, std_pos[2]);

  for (unsigned int i = 0; i < num_particles; i++) {
    
    // calculate new state
    if (fabs(yaw_rate) < EPSILON) {  
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
    
    }else{
      particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
      particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
      particles[i].theta += yaw_rate * delta_t;
    }

    // add noise
    particles[i].x += dist_x(gen);
    particles[i].y += dist_y(gen);
    particles[i].theta += dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  for (unsigned int i = 0; i < observations.size(); i++) {

    // get current observation
    LandmarkObs o = observations[i];

    double min_distance = std::numeric_limits<double>::max();

    // init id of landmark from map placeholder to be associated with the observation
    int map_id = -1;
    
    for (unsigned int j = 0; j < predicted.size(); j++) {
      // get current prediction
      LandmarkObs p = predicted[j];
      
      // get distance between current/predicted landmarks
      // double distance = dist(o.x, o.y, p.x, p.y);
      double distance_sq = (o.x - p.x) * (o.x - p.x) + (o.y - p.y) * (o.y - p.y);

      // find the predicted landmark nearest the current observed landmark
      if (distance_sq < min_distance) {
        min_distance = distance_sq;
        map_id = p.id;
      }
    }

    // set the observation's id to the nearest predicted landmark's id
    observations[i].id = map_id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

  // for each particle...
  for (unsigned int i = 0; i < num_particles; i++) {

    // get the particle x, y coordinates
    double p_x = particles[i].x;
    double p_y = particles[i].y;
    double p_theta = particles[i].theta;

    vector<LandmarkObs> landmarksFiltered;

    // for each map landmark...
    for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {

      // get id and x,y coordinates
      int lm_id = map_landmarks.landmark_list[j].id_i;
      float lm_x = map_landmarks.landmark_list[j].x_f;
      float lm_y = map_landmarks.landmark_list[j].y_f;
      
      
      // only consider landmarks within sensor range of the particle 
       double distance_sq = (lm_x - p_x) * (lm_x - p_x) + (lm_y - p_y) * (lm_y - p_y);
       double sensor_range_sq = sensor_range * sensor_range;

      if (distance_sq < sensor_range_sq) {
          // add prediction to vector
        landmarksFiltered.push_back(LandmarkObs{ lm_id, lm_x, lm_y });
      }
    }

    // create and populate a copy of the list of observations transformed from vehicle coordinates to map coordinates
    vector<LandmarkObs> observations_v;
    
    for (unsigned int j = 0; j < observations.size(); j++) {
      double t_x = cos(p_theta)*observations[j].x - sin(p_theta)*observations[j].y + p_x;
      double t_y = sin(p_theta)*observations[j].x + cos(p_theta)*observations[j].y + p_y;
      observations_v.push_back(LandmarkObs{ observations[j].id, t_x, t_y });
    }

    // perform dataAssociation for the predictions and transformed observations on current particle
    dataAssociation(landmarksFiltered, observations_v);

    /* Prepare associations for debug */
    vector<int> associations;
    vector<double> sense_x;
    vector<double> sense_y;

    for (unsigned int j = 0; j < observations_v.size(); j++) {
      associations.push_back(observations_v[j].id);
      sense_x.push_back(observations_v[j].x);
      sense_y.push_back(observations_v[j].y);
    }

    SetAssociations(particles[i], associations, sense_x, sense_y);

    // reinit weight
    particles[i].weight = 1.0;

    for (unsigned int j = 0; j < observations_v.size(); j++) {
      
      // placeholders for observation and associated prediction coordinates
      double o_x, o_y, pr_x, pr_y;
      o_x = observations_v[j].x;
      o_y = observations_v[j].y;

      int associated_prediction = observations_v[j].id;

      // get the x,y coordinates of the prediction associated with the current observation
      for (unsigned int k = 0; k < landmarksFiltered.size(); k++) {
        if (landmarksFiltered[k].id == associated_prediction) {
          pr_x = landmarksFiltered[k].x;
          pr_y = landmarksFiltered[k].y;
          break;
        }
      }

      // calculate weight for this observation with multivariate Gaussian
      double s_x = std_landmark[0];
      double s_y = std_landmark[1];
      double delta_x = pr_x-o_x;
      double delta_y = pr_y-o_y;

      double obs_w = ( 1/(2*M_PI*s_x*s_y)) * exp( -( delta_x * delta_x / (2*s_x*s_x) + (delta_y * delta_y/(2*s_y*s_y)) ) );

      // product of this obersvation weight with total observations weight
      if (obs_w == 0) {
        particles[i].weight *= EPSILON;
      } else {
        particles[i].weight *= obs_w;
      }
    }
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  vector<double> weights;
  double maxWeight = std::numeric_limits<double>::min();
  for(unsigned int i = 0; i < num_particles; i++) {
    weights.push_back(particles[i].weight);

    if ( particles[i].weight > maxWeight ) {
      maxWeight = particles[i].weight;
    }
  }

  // Creating distributions.
  std::uniform_real_distribution<double> distDouble(0.0, maxWeight);
  std::uniform_int_distribution<int> distInt(0, num_particles - 1);

  // Generating index.
  int index = distInt(gen);

  double beta = 0.0;

  // the wheel
  vector<Particle> resampledParticles;
  for(unsigned int i = 0; i < num_particles; i++) {
    beta += distDouble(gen) * 2.0;

    while( beta > weights[index]) {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    resampledParticles.push_back(particles[index]);
  }

  particles = resampledParticles;

}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  
  //Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();
  
  
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}