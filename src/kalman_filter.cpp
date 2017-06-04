#include <iostream>
#include <math.h>
#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

// 4x4 Identity Matrix
MatrixXd I4 = MatrixXd::Identity(4,4);

// 2*Pi
double pi2 = M_PI * 2;

KalmanFilter::KalmanFilter(): x_(VectorXd(4)), P_(MatrixXd(4,4)),
	F_(MatrixXd(4,4)), H_(MatrixXd(2,4)), R_(MatrixXd(2,2)), 
	Q_(MatrixXd(4,4)) {}

KalmanFilter::~KalmanFilter() {}


void KalmanFilter::Predict() {
  x_ = F_ * x_ /*+ u*/;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  VectorXd y = z - H_ * x_;
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd K = P_ * Ht * Si;
  x_ = x_ + K * y;
    //std::cout << "H_:" << H_ << std::endl;
    //std::cout << "K:" << K << std::endl;
  P_ = (I4 - K * H_) * P_;
}


void KalmanFilter::UpdateEKF(const VectorXd &z, const VectorXd &z1) {
       MatrixXd Hj(3,4);
       // recover state parameters
       double px = z(0);
       double py = z(1);
       double vx = z(2);
       double vy = z(3);
       
       // pre-caulcuate terms
       double c1 = px*px + py*py;
       double c2 = sqrt(c1);
       double c3 = c1*c2;

       // check division by 0
       if (fabs(c1) < 0.0001) {
           //std::cout << "CalculateJacobian got division by zero error!" << std::endl;
           c1 = 0.0001;
       }
    
        if (fabs(c3) < 0.0001) {
            //std::cout << "CalculateJacobian got division by zero error!" << std::endl;
            c3 = 0.0001;
        }

       // Compute the Jacobian matrix
       double item20 = py*(vx*py - vy*px)/c3;
       double item21 = px*(px*vy - py*vx)/c3;
       Hj << (px/c2), (py/c2), 0, 0,
         -(py/c1), (px/c1), 0, 0,
         item20, item21, px/c2, py/c2;
       
       // calc h(x)
       VectorXd h1(3);
       double h1_len = sqrt(x_(0)*x_(0) + x_(1)*x_(1));
       if (h1_len < 0.0001)
           h1_len = 0.0001;
       double rate = (x_(0)*x_(2)+x_(1)*x_(3))/h1_len;
       h1 << h1_len, std::atan2(x_(1), x_(0)), rate;
       VectorXd y = z1 - h1;
        // Make sure y is within Pi and -Pi
        if (y(1) > M_PI) {
            //std::cout << "angle:"<< y(1) ;
            int n = max(int(y(1)/pi2), 1);
            double factor = n*pi2;
            y(1) -= factor;
            //std::cout << " new angle:"<< y(1) << std::endl;
        }
        else if (y(1) < -M_PI) {
            int n = max(int(-y(1)/pi2), 1);
            double factor = n*pi2;
            //std::cout << "angle:"<< y(1) ;
            y(1) += factor;
            //std::cout << " new angle:"<< y(1) << std::endl;
        }

       MatrixXd Ht = Hj.transpose();
       MatrixXd S = Hj * P_ * Ht + R_;
       MatrixXd Si = S.inverse();
       MatrixXd K = P_ * Ht * Si;
       x_ = x_ + K * y;
       P_ = (I4 - K * Hj) * P_;
}
