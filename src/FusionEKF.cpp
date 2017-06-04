#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  // Use noise_ax = 9 and noise_ay = 9 for Q matrix.
  noise_ax_ = 9.0;
  noise_ay_ = 9.0;
 
  // init timestamp
  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
        0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
        0, 0.0009, 0,
        0, 0, 0.09;

  H_laser_ << 1, 0, 0, 0,
             0, 1, 0, 0;
  // state covariance matrix P
  ekf_.P_ << 1, 0, 0, 0,
	     0, 1, 0, 0,
	     0, 0, 1000, 0,
	     0, 0, 0, 1000;
  // measurement matrix
  ekf_.H_ << 1, 0, 0, 0,
	     0, 1, 0, 0;
  // initial transition matrix
  ekf_.F_ << 1, 0, 1, 0,
	     0, 1, 0, 1,
	     0, 0, 1, 0,
	     0, 0, 0, 1;
}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {

  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    // first measurement
    ekf_.x_ = VectorXd(4);

    // init timestamp
    previous_timestamp_ = measurement_pack.timestamp_;

    // init state with the first measurement
    double x = measurement_pack.raw_measurements_[0];
    double y = measurement_pack.raw_measurements_[1];
    double vx = 0.0;
    double vy = 0.0;
    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      // Convert radar from polar to cartesian coordinates
      ekf_.R_ = R_radar_;
      double rho = measurement_pack.raw_measurements_[0];
      double phi = measurement_pack.raw_measurements_[1];
      double rho_dot = measurement_pack.raw_measurements_[2];
      x = rho * cos(phi);
      y = rho * sin(phi);
      vx = rho_dot * cos(phi);
      vy = rho_dot * sin(phi);
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      // Initialize state.
      ekf_.R_ = R_laser_;
      ekf_.H_ = H_laser_;
    }
    ekf_.x_ << x, y, vx, vy;

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

 
  // Update the process noise covariance matrix.
  // Calculate delta time in seconds
  double dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
  previous_timestamp_ = measurement_pack.timestamp_;

  double dt_2 = dt * dt;
  double dt_3 = dt_2 * dt;
  double dt_4 = dt_3 * dt;

  // Update the state transition matrix F according to the new elapsed time
  ekf_.F_(0, 2) = dt;
  ekf_.F_(1, 3) = dt;

  // Set the process covariance matrix Q
  ekf_.Q_ << (dt_4/4*noise_ax_), 0, (dt_3/2*noise_ax_), 0,
	    0, (dt_4/4*noise_ay_), 0, (dt_3/2*noise_ay_),
	    (dt_3/2*noise_ax_), 0, (dt_2*noise_ax_), 0,
	    0, (dt_3/2*noise_ay_), 0, (dt_2*noise_ay_);


  ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  // Use the sensor type to perform the update step.
  // Update the state and covariance matrices.
  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    Eigen::VectorXd v = VectorXd(4);
    double rho = measurement_pack.raw_measurements_[0];
    double phi = measurement_pack.raw_measurements_[1];
    double rho_dot = measurement_pack.raw_measurements_[2];
    v(0) = rho * cos(phi);
    v(1) = rho * sin(phi);
    v(2) = rho_dot * cos(phi);
    v(3) = rho_dot * sin(phi);
    ekf_.R_ = R_radar_;
    // pass both converted values and raw measurements for efficency and accuracy
    ekf_.UpdateEKF(v, measurement_pack.raw_measurements_);
  } else {
    // Laser updates
    ekf_.R_ = R_laser_;
    ekf_.H_ = H_laser_;
    ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
