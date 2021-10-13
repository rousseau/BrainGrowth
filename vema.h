#include <cmath>

class Vector{
public:
	double x, y, z;
	Vector(): x(0.0), y(0.0), z(0.0) {};
	Vector(double ax, double ay, double az): x(ax), y(ay), z(az) {};

	double length(){
		return sqrt(x*x + y*y + z*z);
	}
	double dot(const Vector& b){
		return x*b.x + y*b.y + z*b.z;
	}
	Vector cross(const Vector& b){
		return Vector(y*b.z - z*b.y, z*b.x - x*b.z, x*b.y - y*b.x);
	}
	void normalize(){
		double temp = 1.0/length();
		x *= temp;
		y *= temp;
		z *= temp;
	}
	void clear(){
		x = y = z = 0.0;
	}
	Vector& operator+= (const Vector& b){
		x += b.x;
		y += b.y;
		z += b.z;
		return *this;
	}
	Vector& operator-= (const Vector& b){
		x -= b.x;
		y -= b.y;
		z -= b.z;
		return *this;
	}
	Vector& operator*= (const double& c){
		x *= c;
		y *= c;
		z *= c;
		return *this;
	}
	Vector& operator/= (const double& c){
		x /= c;
		y /= c;
		z /= c;
		return *this;
	}
	Vector operator+ (const Vector& b){
		Vector r = *this;
		return r += b;
	}
	Vector operator- (const Vector& b){
		Vector r = *this;
		return r -= b;
	}
	Vector operator* (const double& c){
		Vector r = *this;
		return r *= c;
	}
	Vector operator/ (const double& c){
		Vector r = *this;
		return r /= c;
	}
};

class Matrix{
public:
  double a, b, c, 
	 d, e, f, 
	 g, h, i;
	 
  Matrix(): a(1.0), b(0.0), c(0.0), 
	    d(0.0), e(1.0), f(0.0), 
	    g(0.0), h(0.0), i(1.0) {};
	    
  Matrix(double aa, double ab, double ac, 
	 double ad, double ae, double af, 
	 double ag, double ah, double ai): 
	 a(aa), b(ab), c(ac), 
	 d(ad), e(ae), f(af), 
	 g(ag), h(ah), i(ai) {};
	 
  Matrix(Vector c1, Vector c2, Vector c3):
	a(c1.x), b(c2.x), c(c3.x),
	d(c1.y), e(c2.y), f(c3.y),
	g(c1.z), h(c2.z), i(c3.z) {};

  double det(){
    return a*e*i - a*f*h - b*d*i + b*f*g + c*d*h - c*e*g;
  }
  double trace(){
    return a + e + i;
  }
  Matrix prod(const Matrix& n) {
    return Matrix( a*n.a+b*n.d+c*n.g, a*n.b+b*n.e+c*n.h, a*n.c+b*n.f+c*n.i, 
		   d*n.a+e*n.d+f*n.g, d*n.b+e*n.e+f*n.h, d*n.c+e*n.f+f*n.i, 
		   g*n.a+h*n.d+i*n.g, g*n.b+h*n.e+i*n.h, g*n.c+h*n.f+i*n.i );
  }
  Vector prod(const Vector& v) {
    return Vector( a*v.x+b*v.y+c*v.z, d*v.x+e*v.y+f*v.z, g*v.x+h*v.y+i*v.z );
  }
  Matrix trans() {
    return Matrix(a, d, g, b, e, h, c, f, i);
  }
  Matrix inv() {
    return Matrix( e*i-f*h, c*h-b*i, b*f-c*e,
		   f*g-d*i, a*i-c*g, c*d-a*f,
		   d*h-e*g, b*g-a*h, a*e-b*d )/det();
  }
  Vector EV() {
	double l1, l2, l3;
	double c1, c0, p, q, phi, t, s;

	c1 = a*e + a*i + e*i - b*b - f*f - c*c;
	c0 = i*b*b + a*f*f + e*c*c - a*e*i - 2.0*c*b*f;
	p = trace()*trace() - 3.0*c1;
	q = trace()*(p - 3.0/2.0*c1) - 27.0/2.0*c0;

	phi = 27.0 * (0.25*c1*c1*(p-c1) + c0*(q + 27.0/4.0*c0));
	phi = 1.0/3.0 * atan2(sqrt(fabs(phi)), q);
	t = sqrt(fabs(p))*cos(phi);
	s = 1.0/sqrt(3.0)*sqrt(fabs(p))*sin(phi);

	l3 = 1.0/3.0*(trace() - t) - s;
	l2 = l3 + 2.0*s;
	l1 = l3 + t + s;	
	return Vector(l1, l2, l3);
  }
  Matrix& operator+= (const Matrix& n){
    a += n.a; b += n.b; c += n.c;
    d += n.d; e += n.e; f += n.f;
    g += n.g; h += n.h; i += n.i;
    return *this;
  }
  Matrix& operator-= (const Matrix& n){
    a -= n.a; b -= n.b; c -= n.c;
    d -= n.d; e -= n.e; f -= n.f;
    g -= n.g; h -= n.h; i -= n.i;
    return *this;
  }
  Matrix& operator*= (const double& z){
    a *= z; b *= z; c *= z;
    d *= z; e *= z; f *= z;
    g *= z; h *= z; i *= z;
    return *this;
  }
  Matrix& operator/= (const double& z){
    a /= z; b /= z; c /= z;
    d /= z; e /= z; f /= z;
    g /= z; h /= z; i /= z;
    return *this;
  }
  Matrix operator+ (const Matrix& n){
    Matrix r = *this;
    return r += n;
  }
  Matrix operator- (const Matrix& n){
    Matrix r = *this;
    return r -= n;
  }
  Matrix operator* (const double& z){
    Matrix r = *this;
    return r *= z;
  }
  Matrix operator/ (const double& z){
    Matrix r = *this;
    return r /= z;
  }
};
