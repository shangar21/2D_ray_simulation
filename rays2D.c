/*
  CSC D18 - Assignment 1 - 2D light propagation

  This is the place where you will be doing most of your work for solving this
  assignment. Before you start working here, you shold have become familiar
  with the data structures and functions available to you from light2D.h, and
  with the way a scene is built in buildScene.c

  Go over each of the functions below, and implement the different components
  of the solution in the sections marked

  /************************
  / TO DO:
  ************************ /

  Do not add or modify code outside these sections.

  Details about what needs to be implemented are described in the comments, as
  well as in the assignment handout. You must read both carefully.

  Starter by: F.J. Estrada, Aug. 2017
*/

/****************************************************************************
 * Uncomment the #define below to enable debug code, add whatever you need
 * to help you debug your program between #ifdef - #endif blocks
 * ************************************************************************/
//#define __DEBUG_MODE
#define FP_TOLERANCE 1e-9

/*****************************************************************************
 * COMPLETE THIS TEXT BOX:
 *
 * 1) Student Name: Sharanshangar Muhunthan
 * 2) Student Name:
 *
 * 1) Student number: 1006291326
 * 2) Student number:
 *
 * 1) muhunth8
 * 2) UtorID
 *
 * We hereby certify that the work contained here is our own
 *
 *  Sharanshangar Muhunthan         _____________________
 * (sign with your name)            (sign with your name)
 ********************************************************************************/
double randomDouble(double min, double max) {
  // From a stack exchange:
  // https://stackoverflow.com/questions/33058848/generate-a-random-double-between-1-and-1
  double range = (max - min);
  double div = RAND_MAX / range;
  return min + (rand() / div);
}

struct point2D randomPointOnUnitCircle() {
  // From a stack exchange:
  // https://stackoverflow.com/questions/5837572/generate-a-random-point-within-a-circle-uniformly/50746409#50746409
  // Modified so that radius is always 1 and not randomized, only randomize the
  // angle instead
  double theta = randomDouble(0, 1) * 2 * PI;
  struct point2D point;

  point.px = cos(theta);
  point.py = sin(theta);

  return point;
}

struct point2D randomPointOnUnitSemiCircle(struct point2D normal) {
  struct point2D point = randomPointOnUnitCircle();
  double dotProd = dot(&point, &normal);
  if (dotProd < FP_TOLERANCE) {
    point.px = -point.px;
    point.py = -point.py;
  }
  return point;
}

double normSquared(struct point2D p) { return p.px * p.px + p.py * p.py; }

struct point2D pointMinus(struct point2D a, struct point2D b) {
  struct point2D res;
  res.px = a.px - b.px;
  res.py = a.py - b.py;
  return res;
}

struct point2D computeLineSegmentEndPoint(struct ray2D ray, double l) {
  // r(l) = p + l d
  // x = px + l dx
  // y = py + l dy
  struct point2D endPoint;
  endPoint.px = ray.p.px + (l * ray.d.px);
  endPoint.py = ray.p.py + (l * ray.d.py);
  return endPoint;
}

double minPositiveQuadraticFormula(double a, double b, double c, double *x1,
                                   double *x2) {
  double det = b * b - 4 * a * c;

  if (det < 0) {
    return -1.0;
  } else {
    det = sqrt(det);
  }

  double plus = (-b + det) / (2 * a);
  double minus = (-b - det) / (2 * a);

  *x1 = plus;
  *x2 = minus;

  if (plus > FP_TOLERANCE && minus > FP_TOLERANCE) {
    return fmin(plus, minus);
  } else if (plus > FP_TOLERANCE) {
    return plus;
  } else if (minus > FP_TOLERANCE) {
    return minus;
  }

  return -1.0; // Both are negative
}

struct point2D getOutwardFacingNormal(struct ray2D ray, struct ray2D wall) {
	// From here https://raytracing.github.io/books/RayTracingInOneWeekend.html, section 6
	// Was used on spheres in source, but just repurposed for circles
  struct point2D wallNormal1;
  struct point2D wallNormal2;

  wallNormal1.px = wall.d.py;
  wallNormal1.py = -wall.d.px;
  normalize(&wallNormal1);

  wallNormal2.px = -wall.d.py;
  wallNormal2.py = wall.d.px;
  normalize(&wallNormal2);

  if (dot(&wallNormal1, &ray.d) < 0) {
    return wallNormal1;
  }

  return wallNormal2;
}

double intersectRayCircle(struct ray2D ray, struct circ2D circ, double *l1,
                          double *l2) {
  // Closest intersection is when t is minimized
  // Consider implicit form of an arbitrary circle ||x - c||^2 - r^2 = 0
  // Plug in r(l) into implcit form and solve for l:
  // ||r(l) - c||^2 - r^2 = 0
  // ||p + ld - c||^2 - r^2 = 0
  // ||(p - c) + ld||^2 - r^2 = 0
  // let v = p - c
  // (v + ld) . (v + ld) - r^2 = 0
  // l^2 ||d||^2 + 2 l d.v + ||v||^2 - r^2 = 0
  // This is a quadratic with a = ||d||^2, b = 2 d.v and c = ||v||^2 - r^2

  struct point2D v = pointMinus(ray.p, circ.c);
  double a = normSquared(ray.d);
  double b = 2 * dot(&ray.d, &v);
  double c = normSquared(v) - (circ.r * circ.r);
  return minPositiveQuadraticFormula(a, b, c, l1, l2);
}

double intersectRayWall(struct ray2D ray, struct wall2D wall) {
  // Closest intersection is when t is minimized
  // let ray's equation be p + l d
  // let wall's equation be q + m v
  // we want to find l d - m v = q - p
  // Note that this leaves us with two equations, and two unknowns, l and m
  // Recall that the inverse of a 2x2 matrix is 1/det(A) * [[d, -b], [-c, a]]
  // Thus we have matrix A = [[a, b], [c, d]] = [[dx, -vx], [dy, -vy]]
  // And we have equation Ax = b => [[dx, -vx], [dy, -vy]] [[l], [m]] = [[qx -
  // px], [qy - py]] To solve a general system [[a, b], [c, d]] [l, m] = [x, y]
  // the solution for l is 1/det(A) (dx - by)
  // the solution for m is 1/det(A) (ay - cx)
  // We will be using this along with the determinant formula ad - bc

  struct ray2D wallRay = wall.w;

  double a = ray.d.px;
  double b = -wallRay.d.px;
  double c = ray.d.py;
  double d = -wallRay.d.py;

  double x = wallRay.p.px - ray.p.px;
  double y = wallRay.p.py - ray.p.py;

  double det = a * d - b * c;

  if (fabs(det) < FP_TOLERANCE)
    return -1;

  double l = (d * x - b * y) / det;
  double m = (a * y - c * x) / det;

  if (l <= 0 || m <= 0 || m > 1)
    return -1;

  return l;
}

struct point2D computeReflection(struct ray2D ray, struct point2D normal) {
  // m = 2(s · n)n −s
  double dotProd = dot(&ray.d, &normal);
  struct point2D m;
  m.px = ray.d.px - 2 * dotProd * normal.px;
  m.py = ray.d.py - 2 * dotProd * normal.py;
  return m;
}

void computeRefraction(struct ray2D ray, double r_idx, struct point2D normal,
                       struct point2D *reflectionDir,
                       struct point2D *refractionDir) {
  struct point2D D = ray.d;
  struct point2D N = normal;
	double n1, n2;

  normalize(&D);
  normalize(&N);

	if (ray.inside_out == 0){
		n1 = 1.0;
		n2 = r_idx;
		ray.inside_out = 1;
	} else {
		n1 = r_idx;
		n2 = 1.0;
		ray.inside_out = 0;
		// Flip normal when inside material
		N.px = -N.px;
		N.py = -N.py;
	}

	double c = -dot(&D, &N);
	double r = n1/n2;

	*reflectionDir = computeReflection(ray, N);
	normalize(reflectionDir);

	if (1 - r*r*(1 - c*c) <= 0){
		return;
	}

	refractionDir->px = r * D.px + (r * c - sqrt(1 - r*r*(1 - c*c))) * N.px;
	refractionDir->py = r * D.py + (r * c - sqrt(1 - r*r*(1 - c*c))) * N.py;
	normalize(refractionDir);

}

struct point2D newDirectionByMaterialType(struct ray2D ray,
                                          struct point2D normal, int type) {
  struct point2D newDirectionObj;
  switch (type) {
  case 0: // reflection
    newDirectionObj = computeReflection(ray, normal);
    break;
  case 1: // scattering
    newDirectionObj = randomPointOnUnitSemiCircle(normal);
  }
  return newDirectionObj;
}

struct ray2D makeLightSourceRay(void) {
  /*
    This function should return a light ray that has its origin at the light
    source, and whose direction depends on the type of light source.

    For point light sources (which emit light in all directions) the direction
     has to be chosen randomly and uniformly over a unit circle (i.e. any
     direction is equally likely)

    For a laser type light source, the direction of the ray is exactly the same
     as the direction of the lightsource.

    Set the colour of the ray to the same colour as the light source, and
     set the inside_outside flag to 0 (we assume light sources are
     outside objects)

    In either case, the direction vector *must be unit length*.
 */

  /************************************************************************
   *  TO DO: Complete this function so that we can sample rays from the
   *         lightsource for the light propagation process.
   ************************************************************************/

  struct ray2D ray;

  // This creates a dummy ray (which won't go anywhere since the direction
  // vector d=[0 0]!. But it's here so you see which data values you have to
  // provide values for given the light source position, and type.
  // ** REPLACE THE CODE BELOW with code that provides a valid ray that
  //    is consistent with the lightsource.

  ray.p.px = 0; // Ray's origin
  ray.p.py = 0;
  ray.d.px = 0; // Ray's direction
  ray.d.py = 0;
  ray.inside_out = 0; // Initially 0 since the ray starts outside an object
  ray.monochromatic =
      0;     // Initially 0 since the ray is white (from lightsource)
  ray.R = 0; // Ray colour in RGB must be the same as the lightsource
  ray.G = 0;
  ray.B = 0;

  ray.p.px = lightsource.l.p.px;
  ray.p.py = lightsource.l.p.py;

  ray.R = lightsource.R;
  ray.G = lightsource.G;
  ray.B = lightsource.B;

  if (lightsource.light_type == 0) {
    // point light
    // generate random direction over unit circle
    struct point2D onUnitCircle = randomPointOnUnitCircle();
    ray.d.px = onUnitCircle.px;
    ray.d.py = onUnitCircle.py;
    //ray.R *= randomDouble(0.0, 1.0);
    //ray.G *= randomDouble(0.0, 1.0);
    //ray.B *= randomDouble(0.0, 1.0);

  } else {
    // laser
    // Does not need to be random, simply just beam from same point in same
    // direction
    ray.d.px = lightsource.l.d.px;
    ray.d.py = lightsource.l.d.py;
    normalize(&ray.d);
  }

  return (ray); // Currently this returns dummy ray
}

void propagateRay(struct ray2D *ray, int depth) {
  if (depth >= max_depth)
    return;

  double lambdaObject;
  struct point2D point;
  struct point2D normal;
  int type;
  double r_idx;

  intersectRay(ray, &point, &normal, &lambdaObject, &type, &r_idx);

  if (lambdaObject > FP_TOLERANCE && lambdaObject != INFINITY) {
    renderRay(&ray->p, &point, ray->R, ray->G, ray->B);

    if (type == 0) { // reflection
      struct point2D reflectionDir = computeReflection(*ray, normal);
      struct ray2D reflectedRay;
      reflectedRay.p = point;
      reflectedRay.d = reflectionDir;
      reflectedRay.R = ray->R;
      reflectedRay.G = ray->G;
      reflectedRay.B = ray->B;
      reflectedRay.inside_out = ray->inside_out;
      normalize(&reflectedRay.d);
      propagateRay(&reflectedRay, ++depth);

    } else if (type == 1) { // scatter
      struct point2D newDirection = randomPointOnUnitSemiCircle(normal);
      struct ray2D scatteredRay;
      scatteredRay.p = point;
      scatteredRay.d = newDirection;
      scatteredRay.R = ray->R;
      scatteredRay.G = ray->G;
      scatteredRay.B = ray->B;
      scatteredRay.inside_out = ray->inside_out;
      normalize(&scatteredRay.d);
      propagateRay(&scatteredRay, ++depth);

    } else if (type == 2) { // refraction
      struct point2D reflectionDir, refractionDir;
      computeRefraction(*ray, r_idx, normal, &reflectionDir, &refractionDir);
			depth++;

      double cosTheta1 = fabs(dot(&ray->d, &normal));
      double R0 = pow((1.0 - r_idx) / (1.0 + r_idx), 2);
      double Rs = R0 + (1.0 - R0) * pow((1.0 - cosTheta1), 5);
      double Rt = 1.0 - Rs;

      struct ray2D reflectedRay;
      reflectedRay.p = point;
      reflectedRay.d = reflectionDir;
      reflectedRay.R = ray->R * Rs;
      reflectedRay.G = ray->G * Rs;
      reflectedRay.B = ray->B * Rs;
      reflectedRay.inside_out = ray->inside_out;
      normalize(&reflectedRay.d);
      propagateRay(&reflectedRay, depth);

      if (refractionDir.px != 0 || refractionDir.py != 0) {
        struct ray2D refractedRay;
        refractedRay.p = point;
        refractedRay.d = refractionDir;
        refractedRay.R = ray->R * Rt;
        refractedRay.G = ray->G * Rt;
        refractedRay.B = ray->B * Rt;
        refractedRay.inside_out = ray->inside_out;
        normalize(&refractedRay.d);
        propagateRay(&refractedRay, depth);
      }
    }
    return; // already found object no need to continue
  }

  for (int i = 0; i < 4; i++) {
    struct wall2D wall = walls[i];
    double lambdaWall = intersectRayWall(*ray, wall);
    if (lambdaWall > 0) {
      struct point2D endPoint = computeLineSegmentEndPoint(*ray, lambdaWall);
      renderRay(&ray->p, &endPoint, ray->R, ray->G, ray->B);
      struct point2D wallNormal = getOutwardFacingNormal(*ray, wall.w);
      struct point2D newDirection =
          newDirectionByMaterialType(*ray, wallNormal, wall.material_type);
      normalize(&newDirection);
      ray->d = newDirection;
      ray->p = endPoint;
      propagateRay(ray, ++depth);
      break;
    }
  }
}

void intersectRay(struct ray2D *ray, struct point2D *p, struct point2D *n,
                  double *lambda, int *type, double *r_idx) {

  *lambda = INFINITY;
  double l1;
  double l2;

  for (int i = 0; i < MAX_OBJECTS; i++) {
    struct circ2D *circle = &objects[i];
    if (circle->r == -1)
      break;
    double tmp = intersectRayCircle(*ray, *circle, &l1, &l2);
    if (tmp > FP_TOLERANCE && tmp < *lambda) {
      *type = circle->material_type;
      *r_idx = circle->r_idx;
      *lambda = tmp;
      *p = computeLineSegmentEndPoint(*ray, *lambda);
      n->px = p->px - circle->c.px;
      n->py = p->py - circle->c.py;
      normalize(n);
    }
  }

  if ((l1 < FP_TOLERANCE && l2 > FP_TOLERANCE) ||
      (l1 > FP_TOLERANCE && l2 < FP_TOLERANCE))
    ray->inside_out =
        1; // if one lambda pos, one neg, then ray is inside medium
}
