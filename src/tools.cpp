#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

using namespace std;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
	VectorXd rmse(4);
	rmse << 0,0,0,0;

	if (estimations.size() != ground_truth.size()
		|| estimations.size() == 0) {
		cout << "Invalid estimation or ground truth data" << endl;
		return rmse;
	}

	for (unsigned int i = 0; i < estimations.size(); i++) {
		VectorXd residual = estimations[i] - ground_truth[i];
		// coef-wise multiplication
		residual = residual.array() * residual.array();
		rmse += residual;
	}

	// calc mean
	rmse = rmse / estimations.size();
    
    // calc the square root
    rmse = rmse.array().sqrt();

	return rmse;
}
