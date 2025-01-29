#include "light2DFrame.h"
#include "buildScene.c"

#define FP_TOLERANCE 1e-9
#define RED_WAVELENGTH 700.0   
#define GREEN_WAVELENGTH 546.1 
#define BLUE_WAVELENGTH 435.8 
//#define __DEBUG_MODE

void hue2RGB(double H, double *R, double *G, double *B) {
  /* Given a HUE value in [0 1], with 0 for deep red and
   * 1 for purple, obtains the corresponding RGB values
   * that give the equivalent colour
   */

  double C, X;

  C = 1.0;
  X = C * (1.0 - fabs(fmod(6.0 * H, 2.0) - 1.0));

  if (H < 1.0 / 6.0) {
    *R = 1.0;
    *G = X;
    *B = 0;
  } else if (H < 2.0 / 6.0) {
    *R = X;
    *G = C;
    *B = 0;
  } else if (H < 3.0 / 6.0) {
    *R = 0;
    *G = C;
    *B = X;
  } else if (H < 4.0 / 6.0) {
    *R = 0;
    *G = X;
    *B = C;
  } else if (H < 5.0 / 6.0) {
    *R = X;
    *G = 0;
    *B = C;
  } else {
    *R = C;
    *G = 0;
    *B = X;
  }
}

void renderObjects(void) {
  /*
   * Useful for debugging - will overlay the objects in the scene onto the
   * final image in green - turn on/off with the debug flag in rays2D.c
   */

  double x, y;
  int xx, yy;

  for (int i = 0; i < MAX_OBJECTS; i++) {
    if (objects[i].r <= 0)
      break;
    for (double ang = 0; ang < 2 * PI; ang += .001) {
      x = objects[i].c.px + (cos(ang) * objects[i].r);
      y = objects[i].c.py + (sin(ang) * objects[i].r);
      x -= W_LEFT;
      y -= W_TOP;
      x = x / (W_RIGHT - W_LEFT);
      y = y / (W_BOTTOM - W_TOP);
      x = x * (sx - 1);
      y = y * (sy - 1);
      xx = (int)round(x);
      yy = (int)round(y);
      if (xx >= 0 && xx < sx && yy >= 0 && yy < sy) {
        *(im + ((xx + (yy * sx)) * 3) + 0) = 0;
        *(im + ((xx + (yy * sx)) * 3) + 1) = 255;
        *(im + ((xx + (yy * sx)) * 3) + 2) = 0;
      }
    }
  }
}

void renderRay(struct point2D *p1, struct point2D *p2, double R, double G,
               double B, struct Frame *frame) {
  /*
    This function renders a ray onto the image from point p1 to point p2, with
    the specified colour R,G,B
  */
  frame->render = true;

  double x1, y1, x2, y2, xt, yt;
  int xx, yy;
  double dx, dy;
  double inc;

  if (p1->px < W_LEFT - TOL || p1->px > W_RIGHT + TOL || p1->py < W_TOP - TOL ||
      p1->py > W_BOTTOM + TOL || p2->px < W_LEFT - TOL ||
      p2->px > W_RIGHT + TOL || p2->py < W_TOP - TOL ||
      p2->py > W_BOTTOM + TOL) {
    fprintf(stderr, "renderRay() - at least one endpoint is outside the image "
                    "bounds, somewhere there's an error...\n");
    fprintf(stderr, "p1=(%f,%f)\n", p1->px, p1->py);
    fprintf(stderr, "p2=(%f,%f)\n", p2->px, p2->py);
  }

  x1 = p1->px - W_LEFT;
  y1 = p1->py - W_TOP;
  x2 = p2->px - W_LEFT;
  y2 = p2->py - W_TOP;

  x1 = x1 / (W_RIGHT - W_LEFT);
  y1 = y1 / (W_BOTTOM - W_TOP);
  x2 = x2 / (W_RIGHT - W_LEFT);
  y2 = y2 / (W_BOTTOM - W_TOP);

  x1 = x1 * (sx - 1);
  y1 = y1 * (sy - 1);
  x2 = x2 * (sx - 1);
  y2 = y2 * (sy - 1);

  dx = x2 - x1;
  dy = y2 - y1;

  if (abs(dx) >= abs(dy)) {
    if (x2 < x1) {
      xt = x1;
      yt = y1;
      x1 = x2;
      y1 = y2;
      x2 = xt;
      y2 = yt;
    }

    yt = y1;
    inc = (y2 - y1) / abs(x2 - x1);
    for (double xt = x1; xt <= x2; xt += 1) {
      xx = (int)round(xt);
      yy = (int)round(yt);
      if (xx >= 0 && xx < sx && yy >= 0 && yy < sy) {
        (*(frame->image + ((xx + (yy * sx)) * 3) + 0)) += R;
        (*(frame->image + ((xx + (yy * sx)) * 3) + 1)) += G;
        (*(frame->image + ((xx + (yy * sx)) * 3) + 2)) += B;
      }
      yt += inc;
    }

  } else {
    if (y2 < y1) {
      xt = x1;
      yt = y1;
      x1 = x2;
      y1 = y2;
      x2 = xt;
      y2 = yt;
    }

    xt = x1;
    inc = (x2 - x1) / abs(y2 - y1);
    for (double yt = y1; yt <= y2; yt += 1) {
      xx = (int)round(xt);
      yy = (int)round(yt);
      if (xx >= 0 && xx < sx && yy >= 0 && yy < sy) {
        (*(frame->image + ((xx + (yy * sx)) * 3) + 0)) += R;
        (*(frame->image + ((xx + (yy * sx)) * 3) + 1)) += G;
        (*(frame->image + ((xx + (yy * sx)) * 3) + 2)) += B;
      }
      xt += inc;
    }
  }
}

void setPixel(double x, double y, double R, double G, double B) {
  /*
   * This function updates the image so that the location corresponding to (x,y)
   * has the specified colour. It handles conversion of coordinates from scene
   * coordinates to image pixel coordinates, and performs a bit of antialiasing
   */

  int xx, yy;
  int ii, jj;
  double W;

  if (R < 0 || G < 0 || B < 0 || R > 1 || G > 1 || B > 1)
    fprintf(stderr, "Invalid RGB colours passed to setPixel() - image will "
                    "have artifacts!\n");

  // Convert to image coordinates
  x -= W_LEFT;
  y -= W_TOP;
  x = x / (W_RIGHT - W_LEFT);
  y = y / (W_BOTTOM - W_TOP);
  x = x * (sx - 1);
  y = y * (sy - 1);

  xx = (int)round(x);
  yy = (int)round(y);

  for (int i = xx - 1; i <= xx + 1; i++)
    for (int j = yy - 1; j <= yy + 1; j++) {
      W = exp(-(((x - i) * (x - i)) + ((y - j) * (y - j))) * .5);
      if (i >= 0 && j >= 0 && i < sx && j < sy) {
        (*(imRGB + ((i + (j * sx)) * 3) + 0)) += W * R;
        (*(imRGB + ((i + (j * sx)) * 3) + 1)) += W * G;
        (*(imRGB + ((i + (j * sx)) * 3) + 2)) += W * B;
      }
    }
}

double dot(struct point2D *p, struct point2D *q) {
  return ((p->px * q->px) + (p->py * q->py));
}

void normalize(struct point2D *d) {
  double l;
  l = d->px * d->px;
  l += (d->py * d->py);
  if (l > 0) {
    l = sqrt(l);
    d->px = d->px / l;
    d->py = d->py / l;
  }
}

void addCirc(struct point2D *c, double r, int type, double r_idx) {
  // This adds an object to the object array. Parameters specify
  // the circle's center c, the radius r, type of material, and
  // index of refraction (only meaningful for transparent objects)
  static int num_obj = 0;

  if (num_obj >= MAX_OBJECTS) {
    fprintf(stderr, "List of objects is full!\n");
    return;
  }
  objects[num_obj].c = *c;
  objects[num_obj].r = r;
  objects[num_obj].material_type = type;
  objects[num_obj].r_idx = r_idx;
  num_obj++;
}

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

  if (ray.inside_out == 0) {
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
  double r = n1 / n2;

  *reflectionDir = computeReflection(ray, N);
  normalize(reflectionDir);

  if (1 - r * r * (1 - c * c) <= 0) {
    return;
  }

  refractionDir->px = r * D.px + (r * c - sqrt(1 - r * r * (1 - c * c))) * N.px;
  refractionDir->py = r * D.py + (r * c - sqrt(1 - r * r * (1 - c * c))) * N.py;
  normalize(refractionDir);
}

struct point2D newDirectionByMaterialType(struct ray2D ray,
                                          struct point2D normal, int type) {
  struct point2D newDirectionObj;
  switch (type) {
  case 0: // reflection
    printf("Reflection\n");
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
    ray.R *= randomDouble(0.0, 1.0);
    ray.G *= randomDouble(0.0, 1.0);
    ray.B *= randomDouble(0.0, 1.0);

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

void propagateRayFrame(struct ray2D *ray, int depth, int frameIdx) {
  if (depth >= max_depth)
    return;

  if (ray->R < TOL && ray->G < TOL && ray->B < TOL)
    return;

  double lambdaObject;
  struct point2D point;
  struct point2D normal;
  int type;
  double r_idx;
  int endIdx;

  intersectRay(ray, &point, &normal, &lambdaObject, &type, &r_idx);

  if (lambdaObject > FP_TOLERANCE && lambdaObject != INFINITY) {
    endIdx =
        renderRayFrame(*ray, lambdaObject, ray->R, ray->G, ray->B, frameIdx);

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
      propagateRayFrame(&reflectedRay, ++depth, endIdx);

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
      propagateRayFrame(&scatteredRay, ++depth, endIdx);

    } else if (type == 2) { // refraction
      struct point2D reflectionDir, refractionDir;
      computeRefraction(*ray, r_idx, normal, &reflectionDir, &refractionDir);

      double cos_theta1 = fabs(dot(&ray->d, &normal));
      double R0 = pow((1.0 - r_idx) / (1.0 + r_idx), 2);
      double Rs = R0 + (1.0 - R0) * pow((1.0 - cos_theta1), 5);
      double Rt = 1.0 - Rs;

      struct ray2D reflectedRay;
      reflectedRay.p = point;
      reflectedRay.d = reflectionDir;
      reflectedRay.R = ray->R * Rs;
      reflectedRay.G = ray->G * Rs;
      reflectedRay.B = ray->B * Rs;
      reflectedRay.inside_out = ray->inside_out;
      normalize(&reflectedRay.d);
      propagateRayFrame(&reflectedRay, depth + 1, endIdx);

      if (refractionDir.px != 0 || refractionDir.py != 0) {
        struct ray2D refractedRay;
        refractedRay.p = point;
        refractedRay.d = refractionDir;
        refractedRay.R = ray->R * Rt;
        refractedRay.G = ray->G * Rt;
        refractedRay.B = ray->B * Rt;
        refractedRay.inside_out = ray->inside_out;
        normalize(&refractedRay.d);
        propagateRayFrame(&refractedRay, depth + 1, endIdx);
      }
    }
    return; // already found object no need to continue
  }

  for (int i = 0; i < 4; i++) {
    struct wall2D wall = walls[i];
    double lambdaWall = intersectRayWall(*ray, wall);
    if (lambdaWall > 0) {
      struct point2D endPoint = computeLineSegmentEndPoint(*ray, lambdaWall);
      endIdx =
          renderRayFrame(*ray, lambdaWall, ray->R, ray->G, ray->B, frameIdx);
      struct point2D wallNormal = getOutwardFacingNormal(*ray, wall.w);
      struct point2D newDirection =
          newDirectionByMaterialType(*ray, wallNormal, wall.material_type);
      normalize(&newDirection);
      ray->d = newDirection;
      ray->p = endPoint;
      propagateRayFrame(ray, ++depth, endIdx);
      break;
    }
  }
}

int renderRayFrame(struct ray2D ray, double lambda, double R, double G,
                   double B, int startIdx) {
  int i;
  double frameLambda = 0.0;
  struct point2D startPoint = ray.p;
  for (i = startIdx; frameLambda < lambda; i++) {
    frameLambda = frameLambda + (FRAME_STEP * (i - startIdx));
    struct point2D endPoint = computeLineSegmentEndPoint(ray, frameLambda);
    struct Frame frame = frameArray.frames[i];
    if (!frame.render)
      addToDynamicFrameArray(&frameArray, sx, sy);
    renderRay(&startPoint, &endPoint, R, G, B, &frame);
		frameArray.frames[i].render = true;
    startPoint = endPoint;
  }
  return i - 1;
}

int main(int argc, char *argv[]) {
  initDynamicFrameArray(&frameArray, 3000, 480, 480);
	printf("Allocated frame array!\n");
  struct ray2D ray;
  struct point2D p, d;
  double mx, mi, rng;
  FILE *f;

  // Parse command line arguments and validate their range
  if (argc < 5) {
    fprintf(stderr, "USAGE: light2D  sx   sy   num_samples   max_depth\n");
    fprintf(stderr, "  sx, sy - image resolution in pixels (in [256 4096])\n");
    fprintf(stderr, "  num_samples - Number of light rays to propagate  (in [1 "
                    "10,000,000])\n");
    fprintf(stderr, "  max_depth - Maximum recursion depth (in [1 25])\n");
    exit(0);
  }
	if (argc == 6) {
		FRAME_STEP = atof(argv[5]);
	}
  sx = atoi(argv[1]);
  sy = atoi(argv[2]);
  num_rays = atoi(argv[3]);
  max_depth = atoi(argv[4]);
  if (sx < 256 || sy < 256 || sx > 4096 || sy > 4096 || num_rays < 1 ||
      num_rays > 10000000 || max_depth < 1 || max_depth > 25) {
    fprintf(stderr, "USAGE: light2D  sx   sy   num_samples   max_depth\n");
    fprintf(stderr, "  sx, sy - image resolution in pixels (in [256 4096])\n");
    fprintf(stderr, "  num_samples - Number of light rays to propagate  (in [1 "
                    "10,000,000])\n");
    fprintf(stderr, "  max_depth - Maximum recursion depth (in [1 25])\n");
    exit(0);
  }
  fprintf(stderr, "Working with:\n");
  fprintf(stderr, "Image size (%d, %d)\n", sx, sy);
  fprintf(stderr, "Number of samples: %d\n", num_rays);
  fprintf(stderr, "Max. recursion depth: %d\n", max_depth);

  // Initialize a blank image for our render
  imRGB = (double *)calloc(sx * sy * 3, sizeof(double));
  if (imRGB == NULL) {
    fprintf(stderr,
            "Out of memory?! is this a Commodore 64 you're running on???\n");
    exit(0);
  }

  // Reset walls and objects arrays
  memset(&objects[0], 0, MAX_OBJECTS * sizeof(struct circ2D));
  memset(&walls[0], 0, 4 * sizeof(struct ray2D));
  for (int i = 0; i < MAX_OBJECTS; i++)
    objects[i].r = -1; // Make sure we can tell any objects
                       // not added by buildScene() have
                       // negative radius!

  // Initialize the walls, scene objects, and lightsource
  buildWalls();
  buildScene();

  // Set image width for coordinate conversion
  w_x = (W_RIGHT - W_LEFT);
  w_y = (W_BOTTOM - W_TOP);

  // READY - The loop below will generate rays in a way that suits the type of
  // lightsource defined in buildScene.c, and will initiate the propagation
  // process.

#pragma omp parallel for schedule(dynamic, 32) private(ray)
  for (int i = 0; i < num_rays; i++) {
    if (num_rays > 10)
      if (i % (num_rays / 10) == 0)
        fprintf(stderr, "Progress=%f\n", (double)i / (double)(num_rays));
#ifdef __DEBUG_MODE
    fprintf(stderr, "Propagating ray %d of %d\n", i, num_rays);
#endif
    ray = makeLightSourceRay();
#ifdef __DEBUG_MODE
    fprintf(stderr, "Ray is at (%f,%f), direction (%f, %f)\n", ray.p.px,
            ray.p.py, ray.d.px, ray.d.py);
#endif
    propagateRayFrame(&ray, 0, 0);
  }

  // Done with light propagation. Process the image array to create a final
  // rendered image

  // First adjust gamma (simple log transform)
  for (int frameIdx = 0; frameIdx < frameArray.size; frameIdx++) {
    imRGB = frameArray.frames[frameIdx].image;
    bool render = frameArray.frames[frameIdx].render;
    if (render) {
      for (int i = 0; i < sx * sy * 3; i++)
        *(imRGB + i) = log((*(imRGB + i)) + 1.5);

      im = (unsigned char *)calloc(sx * sy * 3, sizeof(unsigned char));
      mx = -1;
      mi = 10e15;
      for (int i = 0; i < sx * sy * 3; i++) {
        if (*(imRGB + i) < mi)
          mi = *(imRGB + i);
        if (*(imRGB + i) > mx)
          mx = *(imRGB + i);
      }
      rng = mx - mi;
      fprintf(stderr, "Image range: mi=%f, mx=%f, range=%f\n", mi, mx, rng);

      for (int i = 0; i < sx * sy * 3; i++)
        *(im + i) = (unsigned char)(255.0 * ((*(imRGB + i) - mi) / rng));

#ifdef __DEBUG_MODE
      renderObjects();
#endif

      char fileName[30];
      sprintf(fileName, "frames/frame_%04d.ppm", frameIdx);

      f = fopen(fileName, "w");
      if (f != NULL) {
        fprintf(f, "P6\n");
        fprintf(f, "# Output from Light2D.c\n");
        fprintf(f, "%d %d\n", sx, sy);
        fprintf(f, "255\n");
        fwrite(im, sx * sy * 3 * sizeof(unsigned char), 1, f);
        fclose(f);
      } else
        fprintf(stderr, "Can not create output image file\n");
    }
  }

  // Release resources
  freeDynamicFrameArray(&frameArray);
  free(im);
}
