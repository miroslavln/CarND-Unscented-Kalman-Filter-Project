#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
    // if this is false, laser measurements will be ignored (except during init)
    use_laser_ = true;

    // if this is false, radar measurements will be ignored (except during init)
    use_radar_ = true;

    n_x_ = 5;

    // initial state vector
    x_ = VectorXd::Zero(n_x_);

    // initial covariance matrix
    P_ = MatrixXd::Identity(n_x_, n_x_);

    // Process noise standard deviation longitudinal acceleration in m/s^2
    std_a_ = 2.5;

    // Process noise standard deviation yaw acceleration in rad/s^2
    std_yawdd_ = 0.8;

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

    is_initialized_ = false;

    n_aug_ = 7;

    lambda_ = 3 - n_aug_;

    num_points_ = 2 * n_aug_ + 1;

    Xsig_pred_ = MatrixXd::Zero(n_x_, num_points_);

    weights_ = VectorXd::Zero(num_points_);

    weights_(0) = lambda_ / (lambda_ + n_aug_);
    for (int i = 1; i < num_points_; i++) {
        weights_(i) = 1 / (2 * (lambda_ + n_aug_));
    }

}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
    double dt = (meas_package.timestamp_ - time_us_) / 1000000.0;
    time_us_ = meas_package.timestamp_;

    if (!is_initialized_) {
        Initialize(meas_package);
        is_initialized_ = true;
        return;
    }

    while (dt > 0.1) {
        const double step = 0.05;
        Prediction(step);
        dt -= step;
    }

    Prediction(dt);

    if (meas_package.sensor_type_ == MeasurementPackage::SensorType::RADAR) {
        UpdateRadar(meas_package);

    } else {
        UpdateLidar(meas_package);
    }
}

void UKF::Initialize(const MeasurementPackage &meas_package) {
    if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
        x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0, 0, 0;
    } else {
        double rho = meas_package.raw_measurements_[0];
        double phi = meas_package.raw_measurements_[1];
        x_ << rho * cos(phi), rho * sin(phi), 0, 0, 0;
    }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
    MatrixXd Xsig_aug = CreateAugmentedSigmaPoints();
    PredictSigmaPoints(Xsig_aug, delta_t);
    PredictState();
    PredictStateCovariance();
}

void UKF::PredictStateCovariance() {
    P_.fill(0);
    for (int i = 0; i < num_points_; i++) {
        VectorXd x_diff = Xsig_pred_.col(i) - Xsig_pred_.col(0);
        //angle normalization

        while (x_diff(3) > M_PI) x_diff(3) -= 2. * M_PI;
        while (x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;

        P_ += (x_diff * x_diff.transpose()) * weights_(i);
    }
}

void UKF::PredictState() {
    x_.fill(0);
    for (int i = 0; i < num_points_; i++) {
        x_ += weights_(i) * Xsig_pred_.col(i);
    }
}

void UKF::PredictSigmaPoints(const MatrixXd &Xsig_aug, double delta_t) {
    for (int i = 0; i < num_points_; ++i) {
        VectorXd col = Xsig_aug.col(i);
        double px = col[0];
        double py = col[1];
        double v = col[2];
        double phi = col[3];
        double phi_dot = col[4];
        double noise_a = col[5];
        double noise_phi = col[6];

        double px_p, py_p;
        if (fabs(phi_dot) > 0.001) {
            px_p = px + v / phi_dot * (sin(phi + phi_dot * delta_t) - sin(phi));
            py_p = py + v / phi_dot * (cos(phi) - cos(phi + phi_dot * delta_t));
        } else {
            px_p = px + v * cos(phi) * delta_t;
            py_p = py + v * sin(phi) * delta_t;
        }
        double v_p = v;
        double phi_p = phi + phi_dot * delta_t;
        double phi_dot_p = phi_dot;

        px_p += 0.5 * noise_a * delta_t * delta_t * cos(phi);
        py_p += 0.5 * noise_a * delta_t * delta_t * sin(phi);
        v_p += delta_t * noise_a;
        phi_p += 0.5 * delta_t * delta_t * noise_phi;
        phi_dot_p += delta_t * noise_phi;

        VectorXd pred(n_x_);
        pred << px_p, py_p, v_p, phi_p, phi_dot_p;

        Xsig_pred_.col(i) = pred;
    }
}

MatrixXd UKF::CreateAugmentedSigmaPoints() const {
    MatrixXd Xsig_aug = MatrixXd::Zero(n_aug_, num_points_);

    VectorXd x_aug = VectorXd::Zero(n_aug_);
    MatrixXd P_aug = MatrixXd::Zero(n_aug_, n_aug_);
    x_aug.head(n_x_) = x_;

    //create augmented covariance matrix
    P_aug.topLeftCorner(n_x_, n_x_) = P_;
    MatrixXd Q(2, 2);
    Q << std_a_ * std_a_, 0,
            0, std_yawdd_ * std_yawdd_;
    P_aug.bottomRightCorner(2, 2) = Q;
    //create square root matrix
    MatrixXd A = P_aug.llt().matrixL();
    //create augmented sigma points
    Xsig_aug.col(0) = x_aug;
    Xsig_aug.block(0, 1, n_aug_, n_aug_) = (A * sqrt(lambda_ + n_aug_)).colwise() + x_aug;
    Xsig_aug.block(0, n_aug_ + 1, n_aug_, n_aug_) = (A * -sqrt(lambda_ + n_aug_)).colwise() + x_aug;

    return Xsig_aug;
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
    int n_z = 2;

    MatrixXd Zsig = TransformSigmaPointsToLazerMeasurements();

    MatrixXd R(n_z, n_z);
    R << std_laspx_ * std_laspx_, 0,
            0, std_laspy_ * std_laspy_;

    VectorXd z_pred = CalculateMeasurementPrediction(n_z, Zsig);
    MatrixXd S = CalculateMeasurementCovariance(n_z, Zsig, R, z_pred);
    MatrixXd Tc = CalculateCrossCorrelation(n_z, Zsig, z_pred);
    VectorXd z_diff = GetResidualDifference(meas_package, z_pred);
    UpdateStateMeanAndConvariance(S, Tc, z_diff);

    NIS_laser_ = z_diff.transpose() * S.inverse() * z_diff;
}

MatrixXd UKF::TransformSigmaPointsToLazerMeasurements() const {
    MatrixXd Zsig = MatrixXd::Zero(2, num_points_);
    for (int i = 0; i < num_points_; i++) {
        double px = Xsig_pred_(0, i);
        double py = Xsig_pred_(1, i);
        Zsig(0, i) = px;
        Zsig(1, i) = py;
    }
    return Zsig;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
    int n_z = 3;

    MatrixXd Zsig = TransformSigmaPointsToRadarMeasurements();

    MatrixXd R(n_z, n_z);
    R << std_radr_ * std_radr_, 0, 0,
            0, std_radphi_ * std_radphi_, 0,
            0, 0, std_radrd_ * std_radrd_;

    VectorXd z_pred = CalculateMeasurementPrediction(n_z, Zsig);
    MatrixXd S = CalculateMeasurementCovariance(n_z, Zsig, R, z_pred);
    MatrixXd Tc = CalculateCrossCorrelation(n_z, Zsig, z_pred);
    VectorXd z_diff = GetResidualDifference(meas_package, z_pred);
    UpdateStateMeanAndConvariance(S, Tc, z_diff);

    NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;
}

MatrixXd UKF::TransformSigmaPointsToRadarMeasurements() const {
    MatrixXd Zsig = MatrixXd::Zero(3, num_points_);
    for (int i = 0; i < num_points_; i++) {
        double px = Xsig_pred_(0, i);
        double py = Xsig_pred_(1, i);
        double v = Xsig_pred_(2, i);
        double phi = Xsig_pred_(3, i);
        double phi_dot = Xsig_pred_(4, i);

        if (px == 0 && py == 0) {
            Zsig(0, i) = 0;
            Zsig(1, i) = 0;
            Zsig(2, i) = 0;
        } else {
            Zsig(0, i) = sqrt(px * px + py * py);
            Zsig(1, i) = atan2(py, px);
            Zsig(2, i) = (px * cos(phi) * v + py * sin(phi) * v) / Zsig(0, i);
        }
    }
    return Zsig;
}


VectorXd UKF::GetResidualDifference(const MeasurementPackage &meas_package, const VectorXd &z_pred) const {
    VectorXd z_diff = meas_package.raw_measurements_ - z_pred;

    while (z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
    while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;
    return z_diff;
}

void UKF::UpdateStateMeanAndConvariance(const MatrixXd &S, const MatrixXd &Tc, const VectorXd &z_diff) {
    MatrixXd K = Tc * S.inverse();
    x_ += K * z_diff;
    P_ -= K * S * K.transpose();
}

MatrixXd UKF::CalculateCrossCorrelation(int n_z, const MatrixXd &Zsig, const VectorXd &z_pred) const {
    MatrixXd Tc = MatrixXd::Zero(n_x_, n_z);
    for (int i = 0; i <num_points_; i++) {
        VectorXd x_diff = Xsig_pred_.col(i) - x_;
        while (x_diff(3) > M_PI) x_diff(3) -= 2. * M_PI;
        while (x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;

        VectorXd z_diff = Zsig.col(i) - z_pred;
        while (z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
        while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

        Tc += weights_(i) * x_diff * z_diff.transpose();
    }
    return Tc;
}

MatrixXd UKF::CalculateMeasurementCovariance(int n_z, const MatrixXd &Zsig, const MatrixXd &R,
                                             const VectorXd &z_pred) const {//calculate measurement covariance matrix S
    MatrixXd S = MatrixXd::Zero(n_z, n_z);
    for (int i = 0; i < num_points_; i++) {
        VectorXd z_diff = Zsig.col(i) - z_pred;

        while (z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
        while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

        S += weights_(i) * z_diff * z_diff.transpose();
    }

    S = S + R;
    return S;
}

VectorXd UKF::CalculateMeasurementPrediction(int n_z, const MatrixXd &Zsig) const {
    VectorXd z_pred = VectorXd::Zero(n_z);
    for (int i = 0; i < num_points_; i++) {
        z_pred += Zsig.col(i) * weights_(i);
    }
    return z_pred;
}