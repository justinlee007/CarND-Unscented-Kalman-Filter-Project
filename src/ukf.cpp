#include "ukf.h"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  is_initialized_ = false;

  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // State dimension
  n_x_ = 5;

  // Augmented state dimension
  n_aug_ = n_x_ + 2;

  // Sigma point spreading parameter
  lambda_ = 3 - n_aug_;

  // initial state vector
  x_ = VectorXd(n_x_);

  // initial covariance matrix
  P_ = MatrixXd(n_x_, n_x_);
  P_ << 1, 0, 0, 0, 0,
      0, 1, 0, 0, 0,
      0, 0, 1, 0, 0,
      0, 0, 0, 1, 0,
      0, 0, 0, 0, 1;

  // Augmented sigma point matrix
  Xsig_aug_ = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  // predicted sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  // Weights of sigma points
  weights_ = VectorXd(2 * n_aug_ + 1);

  // set weights
  double weight_0 = lambda_ / (lambda_ + n_aug_);
  weights_(0) = weight_0;
  for (int i = 1; i < 2 * n_aug_ + 1; i++) {
    double weight = 0.5 / (n_aug_ + lambda_);
    weights_(i) = weight;
  }

  // time when the state is true, in us
  time_us_;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 0.2;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.5;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  // the current NIS for radar
  NIS_radar_;

  // the current NIS for laser
  NIS_laser_;

}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR && !use_radar_) {
    return;
  } else if (meas_package.sensor_type_ == MeasurementPackage::LASER && !use_laser_) {
    return;
  }

  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    // first measurement
    float x = 0.0;
    float y = 0.0;
    float xdot = 0.0001;
    float ydot = 0.0001;
    float v = 0.0;
    float orientation = 0.0;
    float orientation_dot = 0.0;

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      // Convert radar from polar to cartesian coordinates and initialize state.
      float rho = meas_package.raw_measurements_[0];
      float phi = meas_package.raw_measurements_[1];
      float rhodot = meas_package.raw_measurements_[2];

      x = rho * cos(phi);
      y = rho * sin(phi);
      xdot = rhodot * cos(phi);
      ydot = rhodot * sin(phi);
      v = sqrt(pow(xdot, 2) + pow(ydot, 2));
      orientation = rho;
      orientation_dot = rhodot;
    } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      // Initialize state.
      x = meas_package.raw_measurements_[0];
      y = meas_package.raw_measurements_[1];
      v = sqrt(pow(xdot, 2) + pow(ydot, 2));
    }

    if (x != 0.0 && y != 0.0) {
      x_ << x, y, v, orientation, orientation_dot;
    } else {
      x_ << 0.0001, 0.0001, 0.1, 0.1, 0.1;
    }

    // done initializing, no need to predict or update
    previous_timestamp_ = meas_package.timestamp_;
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
  *  Prediction
  ****************************************************************************/

  // compute the time elapsed between the current and previous measurements
  float delta_t = (meas_package.timestamp_ - previous_timestamp_) / 1000000.0F; // dt - expressed in seconds
  previous_timestamp_ = meas_package.timestamp_;

  AugmentedSigmaPoints();

  SigmaPointPrediction(delta_t);

  PredictMeanAndCovariance();

  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    PredictRadarMeasurement();
  } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
    PredictLidarMeasurement();
  }
  /*****************************************************************************
  *  Update
  ****************************************************************************/

  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    UpdateRadar(meas_package);
  } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
    UpdateLidar(meas_package);
  }

}

void UKF::AugmentedSigmaPoints() {
  // create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);
  x_aug.head(n_x_) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;

  // create augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  P_aug.fill(0.0);
  P_aug.topLeftCorner(5, 5) = P_;
  P_aug(5, 5) = pow(std_a_, 2);
  P_aug(6, 6) = pow(std_yawdd_, 2);

  // create square root matrix
  MatrixXd L = P_aug.llt().matrixL();

  // create augmented sigma points
  Xsig_aug_.col(0) = x_aug;
  for (int i = 0; i < n_aug_; i++) {
    Xsig_aug_.col(i + 1) = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
    Xsig_aug_.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
  }

  // print result
  // std::cout << "Xsig_aug_ = " << std::endl << Xsig_aug_ << std::endl;
}

void UKF::SigmaPointPrediction(double delta_t) {
  // predict sigma points
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    // extract values for better readability
    double p_x = Xsig_aug_(0, i);
    double p_y = Xsig_aug_(1, i);
    double v = Xsig_aug_(2, i);
    double yaw = Xsig_aug_(3, i);
    double yawd = Xsig_aug_(4, i);
    double nu_a = Xsig_aug_(5, i);
    double nu_yawdd = Xsig_aug_(6, i);

    // predicted state values
    double px_p, py_p;

    // avoid division by zero
    if (fabs(yawd) > 0.001) {
      px_p = p_x + v / yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
      py_p = p_y + v / yawd * (cos(yaw) - cos(yaw + yawd * delta_t));
    } else {
      px_p = p_x + v * delta_t * cos(yaw);
      py_p = p_y + v * delta_t * sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd * delta_t;
    double yawd_p = yawd;

    // add noise
    px_p = px_p + 0.5 * nu_a * delta_t * delta_t * cos(yaw);
    py_p = py_p + 0.5 * nu_a * delta_t * delta_t * sin(yaw);
    v_p = v_p + nu_a * delta_t;

    yaw_p = yaw_p + 0.5 * nu_yawdd * delta_t * delta_t;
    yawd_p = yawd_p + nu_yawdd * delta_t;

    // write predicted sigma point into right column
    Xsig_pred_(0, i) = px_p;
    Xsig_pred_(1, i) = py_p;
    Xsig_pred_(2, i) = v_p;
    Xsig_pred_(3, i) = yaw_p;
    Xsig_pred_(4, i) = yawd_p;
  }

  // print result
  // std::cout << "Xsig_pred_ = " << std::endl << Xsig_pred_ << std::endl;
}

void UKF::PredictMeanAndCovariance() {
  // 2n+1 weights
  int total_weights = 2 * n_aug_ + 1;

  // predicted state mean
  x_.fill(0.0);

  // iterate over sigma points
  for (int i = 0; i < total_weights; i++) {
    x_ = x_ + weights_(i) * Xsig_pred_.col(i);
  }

  // predicted state covariance matrix
  P_.fill(0.0);

  // iterate over sigma points
  for (int i = 0; i < total_weights; i++) {

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    // angle normalization
    while (x_diff(3) > M_PI) {
      x_diff(3) -= 2. * M_PI;
    }

    while (x_diff(3) < -M_PI) {
      x_diff(3) += 2. * M_PI;
    }

    P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
  }

  // print result
  // std::cout << "Predicted state" << std::endl;
  // std::cout << x_ << std::endl;
  // std::cout << "Predicted covariance matrix" << std::endl;
  // std::cout << P_ << std::endl;
}

void UKF::PredictLidarMeasurement() {
  // set measurement dimension, lidar can measure x and y
  int n_z = 2;

  // 2n+1 simga points
  int total_points = 2 * n_aug_ + 1;

  // create matrix for sigma points in measurement space
  Zsig_ = MatrixXd(n_z, total_points);

  // transform sigma points into measurement space
  for (int i = 0; i < total_points; i++) {

    // extract values for better readibility
    double p_x = Xsig_pred_(0, i);
    double p_y = Xsig_pred_(1, i);
    double v = Xsig_pred_(2, i);
    double yaw = Xsig_pred_(3, i);

    // measurement model
    Zsig_(0, i) = p_x;
    Zsig_(1, i) = p_y;
  }

  //mean predicted measurement
  z_pred_ = VectorXd(n_z);
  z_pred_.fill(0.0);
  for (int i = 0; i < total_points; i++) {
    z_pred_ = z_pred_ + weights_(i) * Zsig_.col(i);
  }

  //measurement covariance matrix S
  S_ = MatrixXd(n_z, n_z);
  S_.fill(0.0);
  for (int i = 0; i < total_points; i++) {
    //residual
    VectorXd z_diff = Zsig_.col(i) - z_pred_;

    S_ = S_ + weights_(i) * z_diff * z_diff.transpose();
  }

  // add measurement noise covariance matrix
  MatrixXd R = MatrixXd(n_z, n_z);
  R << std_laspx_ * std_laspx_, 0,
      0, std_laspy_ * std_laspy_;
  S_ = S_ + R;

  // print result
  // std::cout << "z_pred_: " << std::endl << z_pred_ << std::endl;
  // std::cout << "S_: " << std::endl << S_ << std::endl;
}

void UKF::PredictRadarMeasurement() {
  // set measurement dimension, radar can measure r, phi, and r_dot
  int n_z = 3;

  // 2n+1 simga points
  int total_points = 2 * n_aug_ + 1;

  // create matrix for sigma points in measurement space
  Zsig_ = MatrixXd(n_z, total_points);

  // transform sigma points into measurement space
  for (int i = 0; i < total_points; i++) {

    // extract values for better readability
    double p_x = Xsig_pred_(0, i);
    double p_y = Xsig_pred_(1, i);
    double v = Xsig_pred_(2, i);
    double yaw = Xsig_pred_(3, i);

    double v1 = cos(yaw) * v;
    double v2 = sin(yaw) * v;

    // measurement model
    Zsig_(0, i) = sqrt(p_x * p_x + p_y * p_y); // rho
    Zsig_(1, i) = atan2(p_y, p_x); // phi
    Zsig_(2, i) = (p_x * v1 + p_y * v2) / sqrt(p_x * p_x + p_y * p_y); // rhodot
  }

  // mean predicted measurement
  z_pred_ = VectorXd(n_z);
  z_pred_.fill(0.0);
  for (int i = 0; i < total_points; i++) {
    z_pred_ = z_pred_ + weights_(i) * Zsig_.col(i);
  }

  // measurement covariance matrix S
  S_ = MatrixXd(n_z, n_z);
  S_.fill(0.0);
  for (int i = 0; i < total_points; i++) {
    // residual
    VectorXd z_diff = Zsig_.col(i) - z_pred_;

    // angle normalization
    while (z_diff(1) > M_PI) {
      z_diff(1) -= 2. * M_PI;
    }
    while (z_diff(1) < -M_PI) {
      z_diff(1) += 2. * M_PI;
    }

    S_ = S_ + weights_(i) * z_diff * z_diff.transpose();
  }

  // add measurement noise covariance matrix
  MatrixXd R = MatrixXd(n_z, n_z);
  R << std_radr_ * std_radr_, 0, 0,
      0, std_radphi_ * std_radphi_, 0,
      0, 0, std_radrd_ * std_radrd_;
  S_ = S_ + R;

  // print result
  // std::cout << "z_pred_: " << std::endl << z_pred_ << std::endl;
  // std::cout << "S_: " << std::endl << S_ << std::endl;
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */

  // set measurement dimension, lidar can measure x and y
  int n_z = 2;

  // Create vector for incoming lidar measurement
  VectorXd z = meas_package.raw_measurements_;

  //calculate cross correlation matrix
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
    //residual
    VectorXd z_diff = Zsig_.col(i) - z_pred_;

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  //Kalman gain K;
  MatrixXd K = Tc * S_.inverse();

  //residual
  VectorXd z_diff = z - z_pred_;

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S_ * K.transpose();

  // Calculate NIS
  NIS_laser_ = z_diff.transpose() * S_.inverse() * z_diff;

  // print result
  // std::cout << "Updated state x_: " << std::endl << x_ << std::endl;
  // std::cout << "Updated state covariance P_: " << std::endl << P_ << std::endl;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
  // set measurement dimension, radar can measure r, phi, and r_dot
  int n_z = 3;

  // Create vector for incoming radar measurement
  VectorXd z = meas_package.raw_measurements_;

  // create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);

  // calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

    // residual
    VectorXd z_diff = Zsig_.col(i) - z_pred_;

    // angle normalization
    while (z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
    while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    // angle normalization
    while (x_diff(3) > M_PI) {
      x_diff(3) -= 2. * M_PI;
    }
    while (x_diff(3) < -M_PI) {
      x_diff(3) += 2. * M_PI;
    }

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  // Kalman gain K;
  MatrixXd K = Tc * S_.inverse();

  // residual
  VectorXd z_diff = z - z_pred_;

  // angle normalization
  while (z_diff(1) > M_PI) {
    z_diff(1) -= 2. * M_PI;
  }
  while (z_diff(1) < -M_PI) {
    z_diff(1) += 2. * M_PI;
  }

  // update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S_ * K.transpose();

  // Calculate NIS
  NIS_radar_ = z_diff.transpose() * S_.inverse() * z_diff;

  // print result
  // std::cout << "Updated state x_: " << std::endl << x_ << std::endl;
  // std::cout << "Updated state covariance P_: " << std::endl << P_ << std::endl;
}
