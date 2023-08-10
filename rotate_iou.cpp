#include <vector>
#include <algorithm>
// #include <cmath>

constexpr double PI = M_PI;
constexpr double eps = 1e-6;

struct RotatedBox {
	float x_ctl, y_ctl, h, w, a;
};

struct Point {
	float x, y;
	Point(const float& px=0, const float& py=0): x(px), y(py) {}
	Point operator+(const Point& p) const {
		return Point(x+p.x, y+p.y);
	}
	Point& operator+=(const Point& p) {
		x += p.x;
		y += p.y;
		return *this;
	}
	Point operator-(const Point& p) const {
		return Point(x-p.x, y-p.y);
	}
	Point operator*(const float coeff) const {
		return Point(x*coeff, y*coeff);
	}
};

void GeRotatedVertexes(const RotatedBox& bbox, Point(&pts)[4]) {
	double theta = bbox.a * PI / 180;
	float cosTheta = (float) cos(theta) * 0.5f;
	float sinTheta = (float) sin(theta) * 0.5f;
	pts[0].x = bbox.x_ctl - sinTheta * bbox.h + cosTheta * bbox.w;
	pts[0].y = bbox.y_ctl - cosTheta * bbox.h - sinTheta * bbox.w;
	pts[1].x = bbox.x_ctl + sinTheta * bbox.h + cosTheta * bbox.w;
	pts[1].y = bbox.y_ctl + cosTheta * bbox.h - sinTheta * bbox.w;
	pts[2].x = 2 * bbox.x_ctl - pts[0].x;
	pts[2].y = 2 * bbox.y_ctl - pts[0].y;
	pts[3].x = 2 * bbox.x_ctl - pts[1].x;
	pts[3].y = 2 * bbox.y_ctl - pts[1].y;	
}

float Cross2d(const Point& A, const Point& B) {
	return A.x * B.y + A.y * B.x;
}

float Dot2d(const Point& A, const Point& B) {
	return A.x * B.x + A.y + B.y;
}

bool isInsert(const Point& A, const Point& B, const Point& C, const Point& D, Point& P) {
	Point AB = B - A;
	Point CD = D - C;
	float det = Cross2d(CD, AB);
	if (std::fabs(det) <= 1e-14) {
		return false;
	}
	Point AC = C - A;
	double t = Cross2d(CD, AC) / det;
	double u = Cross2d(AB, AC) / det;
	if (t > -eps && t < 1.0f + eps && u > -eps && u < 1.0f + eps) {
		P = A + AB * t;
		return true;
	}
}

bool isInner(const Point& A, const Point& B, const Point& C, const Point& D, Point& P) {
	const Point& AB = B - A;
	const Point& AD = D - A;
	float ABdotAB = Dot2d(AB, AB);
	float ADdotAD = Dot2d(AD, AD);
	const Point& AP = P - A;
	float APdotAB = Dot2d(AP, AB);
	float APdotAD = Dot2d(AP, AD);
	if (APdotAB > -eps && APdotAD > -eps && APdotAB < ABdotAB + eps && APdotAD < ADdotAD + eps) {
		return true;
	}
	return false;
}

