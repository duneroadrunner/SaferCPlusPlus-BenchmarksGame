/* The Computer Language Benchmarks Game
http://benchmarksgame.alioth.debian.org/

contributed by Mark C. Lewis
modified slightly by Chad Whipkey
converted from java to c++,added sse support, by Branimir Maksimovic
modified by Vaclav Zeman
*/

//define ALIGNAS(x) alignas(x)
#ifdef _MSC_VER
#define ALIGNAS(x) __declspec(align(x))
#else /*_MSC_VER*/
#define ALIGNAS(x) __attribute__((aligned(x)))
#endif /*_MSC_VER*/

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <array>
#include <immintrin.h>

#include <string>
#include <iomanip>
#include <iostream>
#include "msemsearray.h"

static const double PI = 3.141592653589793;
static const double SOLAR_MASS = 4 * PI * PI;
static const double DAYS_PER_YEAR = 365.24;


class Body {

public:
	double x = 0;
	double y = 0;
	double z = 0;
	double filler = 0;
	double vx = 0;
	double vy = 0;
	double vz = 0;
	double mass = 0;

	Body() {}

	static Body jupiter() {
		static Body p;
		p.x = 4.84143144246472090e+00;
		p.y = -1.16032004402742839e+00;
		p.z = -1.03622044471123109e-01;
		p.vx = 1.66007664274403694e-03 * DAYS_PER_YEAR;
		p.vy = 7.69901118419740425e-03 * DAYS_PER_YEAR;
		p.vz = -6.90460016972063023e-05 * DAYS_PER_YEAR;
		p.mass = 9.54791938424326609e-04 * SOLAR_MASS;
		return p;
	}

	static Body saturn() {
		static Body p;
		p.x = 8.34336671824457987e+00;
		p.y = 4.12479856412430479e+00;
		p.z = -4.03523417114321381e-01;
		p.vx = -2.76742510726862411e-03 * DAYS_PER_YEAR;
		p.vy = 4.99852801234917238e-03 * DAYS_PER_YEAR;
		p.vz = 2.30417297573763929e-05 * DAYS_PER_YEAR;
		p.mass = 2.85885980666130812e-04 * SOLAR_MASS;
		return p;
	}

	static Body uranus() {
		static Body p;
		p.x = 1.28943695621391310e+01;
		p.y = -1.51111514016986312e+01;
		p.z = -2.23307578892655734e-01;
		p.vx = 2.96460137564761618e-03 * DAYS_PER_YEAR;
		p.vy = 2.37847173959480950e-03 * DAYS_PER_YEAR;
		p.vz = -2.96589568540237556e-05 * DAYS_PER_YEAR;
		p.mass = 4.36624404335156298e-05 * SOLAR_MASS;
		return p;
	}

	static Body neptune() {
		static Body p;
		p.x = 1.53796971148509165e+01;
		p.y = -2.59193146099879641e+01;
		p.z = 1.79258772950371181e-01;
		p.vx = 2.68067772490389322e-03 * DAYS_PER_YEAR;
		p.vy = 1.62824170038242295e-03 * DAYS_PER_YEAR;
		p.vz = -9.51592254519715870e-05 * DAYS_PER_YEAR;
		p.mass = 5.15138902046611451e-05 * SOLAR_MASS;
		return p;
	}

	static Body sun() {
		static Body p;
		p.mass = SOLAR_MASS;
		return p;
	}

	void offsetMomentum(double px, double py, double pz) {
		vx = -px / SOLAR_MASS;
		vy = -py / SOLAR_MASS;
		vz = -pz / SOLAR_MASS;
	}
};


class NBodySystem {
private:
	mse::msearray<Body, 5> bodies;

public:
	NBodySystem()
		: bodies{ {
				Body::sun(),
				Body::jupiter(),
				Body::saturn(),
				Body::uranus(),
				Body::neptune()
			} }
	{
		double px = 0.0;
		double py = 0.0;
		double pz = 0.0;
		for (mse::msear_size_t i = 0; i < bodies.size(); ++i) {
			px += bodies[i].vx * bodies[i].mass;
			py += bodies[i].vy * bodies[i].mass;
			pz += bodies[i].vz * bodies[i].mass;
		}
		bodies[0].offsetMomentum(px, py, pz);
	}

	void advance(double dt) {
		const mse::msear_size_t N = (bodies.size() - 1)*bodies.size() / 2;
		struct ALIGNAS(16) R {
			double dx = 0;
			double dy = 0;
			double dz = 0;
			double filler = 0;
		};
		//static R r[1000];
		static mse::msearray<R, 1000> r;

		//static ALIGNAS(16) double mag[1000];
		static ALIGNAS(16) mse::msearray<double, 1000> mag;

		for (mse::msear_size_t i = 0, k = 0; i < bodies.size() - 1; ++i) {
			Body& iBody = bodies[i];
			for (mse::msear_size_t j = i + 1; j < bodies.size(); ++j, ++k) {
				r[k].dx = iBody.x - bodies[j].x;
				r[k].dy = iBody.y - bodies[j].y;
				r[k].dz = iBody.z - bodies[j].z;
			}
		}

		for (mse::msear_size_t i = 0; i < N; i += 2) {
			__m128d dx = __m128d();
			__m128d dy = __m128d();
			__m128d dz = __m128d();
			dx = _mm_loadl_pd(dx, &r[i].dx);
			dy = _mm_loadl_pd(dy, &r[i].dy);
			dz = _mm_loadl_pd(dz, &r[i].dz);

			dx = _mm_loadh_pd(dx, &r[i + 1].dx);
			dy = _mm_loadh_pd(dy, &r[i + 1].dy);
			dz = _mm_loadh_pd(dz, &r[i + 1].dz);


			//__m128d dSquared = dx*dx + dy*dy + dz*dz;
			__m128d dSquared = _mm_add_pd(_mm_add_pd(_mm_mul_pd(dx, dx), _mm_mul_pd(dy, dy)), _mm_mul_pd(dz, dz));


			__m128d distance =
				_mm_cvtps_pd(_mm_rsqrt_ps(_mm_cvtpd_ps(dSquared)));
			for (mse::msear_size_t j = 0; j<2; ++j)
			{
				/*
				distance = distance * _mm_set1_pd(1.5) -
				((_mm_set1_pd(0.5) * dSquared) * distance) *
				(distance * distance);
				*/
				distance = _mm_sub_pd(_mm_mul_pd(distance, _mm_set1_pd(1.5)),
					_mm_mul_pd(_mm_mul_pd(_mm_mul_pd(_mm_set1_pd(0.5), dSquared), distance),
						_mm_mul_pd(distance, distance)));
			}

			//__m128d dmag = _mm_set1_pd(dt) / (dSquared)* distance;
			__m128d dmag = _mm_mul_pd(_mm_div_pd(_mm_set1_pd(dt), (dSquared)), distance);
			_mm_store_pd(&(mag[i]), dmag);
		}

		for (mse::msear_size_t i = 0, k = 0; i < bodies.size() - 1; ++i) {
			Body& iBody = bodies[i];
			for (mse::msear_size_t j = i + 1; j < bodies.size(); ++j, ++k) {
				iBody.vx -= r[k].dx * bodies[j].mass * mag[k];
				iBody.vy -= r[k].dy * bodies[j].mass * mag[k];
				iBody.vz -= r[k].dz * bodies[j].mass * mag[k];

				bodies[j].vx += r[k].dx * iBody.mass * mag[k];
				bodies[j].vy += r[k].dy * iBody.mass * mag[k];
				bodies[j].vz += r[k].dz * iBody.mass * mag[k];
			}
		}

		for (mse::msear_size_t i = 0; i < bodies.size(); ++i) {
			bodies[i].x += dt * bodies[i].vx;
			bodies[i].y += dt * bodies[i].vy;
			bodies[i].z += dt * bodies[i].vz;
		}
	}

	double energy() {
		double e = 0.0;

		double dx = 0;
		double dy = 0;
		double dz = 0;
		double distance = 0;
		for (mse::msear_size_t i = 0; i < bodies.size(); ++i) {
			//Body const & iBody = bodies[i];
			e += 0.5 * bodies[i].mass *
				(bodies[i].vx * bodies[i].vx
					+ bodies[i].vy * bodies[i].vy
					+ bodies[i].vz * bodies[i].vz);

			for (mse::msear_size_t j = i + 1; j < bodies.size(); ++j) {
				//Body const & jBody = bodies[j];
				dx = bodies[i].x - bodies[j].x;
				dy = bodies[i].y - bodies[j].y;
				dz = bodies[i].z - bodies[j].z;

				distance = sqrt(dx*dx + dy*dy + dz*dz);
				e -= (bodies[i].mass * bodies[j].mass) / distance;
			}
		}
		return e;
	}
};

int main(int argc, char** argv) {
	int n = 50000000;
	if (2 <= argc) {
		n = std::stoi(argv[1]);
	}

	NBodySystem bodies;
	std::cout.precision(9);
	std::cout.setf(std::ios::fixed, std::ios::floatfield); // floatfield set to fixed
	std::cout << bodies.energy() << '\n';
	for (int i = 0; i < n; ++i)
		bodies.advance(0.01);
	std::cout << bodies.energy() << '\n';

}
