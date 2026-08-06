#version 300 es
precision highp float;
precision highp int;

// Schwarzschild null geodesics, planar Binet reduction: d2u/dphi2 = 3u^2 - u,
// u = 1/r, G = c = M = 1. Same physics as the CPU-validated reference.

out vec4 fragColor;

uniform vec2  uRes;
uniform vec3  uCamPos, uCamFwd, uCamRight, uCamUp;
uniform float uTanHalfFov, uAspect, uRcam;
uniform float uRin, uRout, uExposure, uTime, uSpin;
uniform float uUEscape;
uniform int   uMaxSteps;
uniform int   uGRShade;      // 0 = emissivity only, 1 = full redshift + beaming

const float U_HORIZON = 1.0 / (2.0 * 1.0001);

// ---------------------------------------------------------------- hash / stars

uint hashu(uint x) {
    x ^= x >> 16u; x *= 0x7feb352du;
    x ^= x >> 15u; x *= 0x846ca68bu;
    x ^= x >> 16u; return x;
}
float rnd(uint s) { return float(hashu(s) & 0x00ffffffu) * (1.0 / 16777216.0); }

vec3 starfield(vec3 d) {
    const int NA = 1400, NC = 700;
    float a = (atan(d.y, d.x) + 3.14159265) * (1.0 / 6.2831853);
    float c = (d.z + 1.0) * 0.5;
    float fa = a * float(NA), fc = c * float(NC);
    int ia = int(floor(fa)), ic = int(floor(fc));

    vec3 acc = vec3(0.0);
    for (int dj = -1; dj <= 1; ++dj) {
        int jc = ic + dj;
        if (jc < 0 || jc >= NC) continue;
        for (int di = -1; di <= 1; ++di) {
            int ja = (ia + di + NA) % NA;
            uint seed = uint(ja * 73856093) ^ uint(jc * 19349663);
            if (rnd(seed) > 0.055) continue;

            float sx = float(ja) + rnd(seed ^ 0xa511e9b3u);
            float sy = float(jc) + rnd(seed ^ 0x0e5c1f2du);
            float dx = fa - sx, dy = fc - sy;
            if (dx >  float(NA) * 0.5) dx -= float(NA);
            if (dx < -float(NA) * 0.5) dx += float(NA);
            float d2 = dx * dx + dy * dy;

            float mag  = rnd(seed ^ 0x1b873593u);
            float lum  = 0.02 + 1.6 * mag * mag * mag * mag * mag;
            float sig2 = 0.16 + 0.5 * lum;
            float prof = exp(-d2 / sig2);

            float t = rnd(seed ^ 0x27d4eb2fu);
            vec3 col = (t > 0.82) ? vec3(0.72, 0.80, 1.00)
                                  : vec3(1.00, 0.78 + 0.20 * t, 0.55 + 0.45 * t);
            acc += col * (lum * prof);
        }
    }
    acc += vec3(0.012, 0.011, 0.016) * exp(-abs(d.y) * 6.0);
    return acc;
}

// ---------------------------------------------------------------- integrator

void deriv(float u, float du, out float fu, out float fdu) {
    fu = du; fdu = 3.0 * u * u - u;
}

void rk4(inout float u, inout float du, float h) {
    float a1, b1, a2, b2, a3, b3, a4, b4;
    deriv(u,               du,               a1, b1);
    deriv(u + 0.5*h*a1,    du + 0.5*h*b1,    a2, b2);
    deriv(u + 0.5*h*a2,    du + 0.5*h*b2,    a3, b3);
    deriv(u + h*a3,        du + h*b3,        a4, b4);
    u  += (h / 6.0) * (a1 + 2.0*a2 + 2.0*a3 + a4);
    du += (h / 6.0) * (b1 + 2.0*b2 + 2.0*b3 + b4);
}

float stepSize(float u, float du) {
    float h = 0.05 / (1.0 + 30.0 * u * u);      // refine near the photon sphere
    h = min(h, 0.3 * abs(u) / (abs(du) + 1.0e-30));
    return max(h, 1.0e-6);
}

// ---------------------------------------------------------------- shading

float redshift(float rEm, float LzE) {
    float omega = uSpin * inversesqrt(rEm * rEm * rEm);   // Keplerian
    float num = sqrt(max(1.0 - 3.0 / rEm, 1.0e-6));
    float den = sqrt(max(1.0 - 2.0 / uRcam, 1.0e-6)) * (1.0 - omega * LzE);
    den = (abs(den) < 1.0e-4) ? sign(den) * 1.0e-4 : den;
    return num / den;
}

float emissivity(float r) {
    float x = uRin / r;
    float f = x * x * x * (1.0 - sqrt(x));
    float xp = (6.0 / 7.0) * (6.0 / 7.0);                 // argmax of x^3(1-sqrt x)
    return f / (xp * xp * xp * (1.0 - sqrt(xp)));
}

vec3 tempColour(float t) {
    // Knots placed against the measured T distribution: the receding side
    // (p05..p50 of T) stays deep orange, only the beamed approaching side
    // crosses into white-blue. Wide orange span is what keeps the disk from
    // rendering beige-gray after Reinhard.
    t = clamp(t, 0.0, 1.0);
    vec3 deep = vec3(0.95, 0.18, 0.02);
    vec3 org  = vec3(1.00, 0.45, 0.08);
    vec3 amb  = vec3(1.00, 0.72, 0.30);
    vec3 hot  = vec3(0.88, 0.90, 1.00);
    if (t < 0.30) return mix(deep, org, t / 0.30);
    if (t < 0.65) return mix(org, amb, (t - 0.30) / 0.35);
    return mix(amb, hot, (t - 0.65) / 0.35);
}

vec3 shadeDisk(float r, float az, float LzE) {
    float g  = (uGRShade == 1) ? redshift(r, LzE) : 1.0;
    float em = emissivity(r);
    // g*em^(1/4) spans 0.56..1.11 (5th..95th pct, measured); remap so the
    // median 0.73 sits at the amber part of the ramp, not the pale middle
    float T  = (pow(max(em, 1.0e-6), 0.25) * g - 0.40) * 1.15;
    // pattern advected at the local Keplerian rate: differential rotation
    float pat  = az - uSpin * inversesqrt(r * r * r) * uTime;
    float turb = 0.72 + 0.28 * sin(11.0 * log(r) + 5.0 * pat)
                      * (0.65 + 0.35 * sin(27.0 * log(r) - 9.0 * pat));
    float g2 = g * g;
    return tempColour(T) * (em * g2 * g2 * turb * uExposure);
}

// ---------------------------------------------------------------- trace

vec3 trace(vec3 P, vec3 D) {
    float r0 = length(P);
    vec3  n  = cross(P, D);
    float nn = length(n);
    if (nn < 1.0e-7 * r0)
        return (dot(P, D) < 0.0) ? vec3(0.0) : starfield(normalize(D));
    n /= nn;

    vec3 e1 = P / r0;
    vec3 e2 = normalize(cross(n, e1));
    float dr = dot(D, e1);
    float dt = dot(D, e2);

    float lapse = sqrt(max(0.0, 1.0 - 2.0 / r0));   // static-observer frame
    float u  = 1.0 / r0;
    float du = -lapse * dr / (r0 * dt);

    float b   = inversesqrt(max(du*du + u*u - 2.0*u*u*u, 1.0e-30));
    float LzE = -b * n.z;    // traced ray runs backward along the photon path

    float phi = 0.0;
    float sPrev = e1.z;

    for (int i = 0; i < 1024; ++i) {
        if (i >= uMaxSteps) break;
        float up = u, dup = du, phip = phi;

        float h = stepSize(u, du);
        rk4(u, du, h);
        phi += h;

        if (u < uUEscape) {                          // escaped, u may be < 0
            vec3 rad = cos(phip) * e1 + sin(phip) * e2;
            vec3 tng = -sin(phip) * e1 + cos(phip) * e2;
            vec3 vel = -(dup / (up * up)) * rad + (1.0 / up) * tng;
            return starfield(normalize(vel));
        }

        float s = cos(phi) * e1.z + sin(phi) * e2.z;
        if (sPrev * s < 0.0) {
            float t  = sPrev / (sPrev - s);
            float t2 = t * t, t3 = t2 * t;
            float uc = ( 2.0*t3 - 3.0*t2 + 1.0) * up
                     + (     t3 - 2.0*t2 + t  ) * (h * dup)
                     + (-2.0*t3 + 3.0*t2      ) * u
                     + (     t3 -     t2      ) * (h * du);
            float rc = 1.0 / max(uc, 1.0e-20);
            if (rc > uRin && rc < uRout) {
                float phic = phip + t * h;
                vec3 pos = rc * (cos(phic) * e1 + sin(phic) * e2);
                return shadeDisk(rc, atan(pos.y, pos.x), LzE);
            }
        }
        sPrev = s;

        if (u > U_HORIZON) return vec3(0.0);
    }
    return vec3(0.0);                                // step budget: near b_c
}

void main() {
    float sx = 2.0 * gl_FragCoord.x / uRes.x - 1.0;
    float sy = 2.0 * gl_FragCoord.y / uRes.y - 1.0;
    vec3 D = normalize(uCamFwd
                     + uCamRight * (sx * uTanHalfFov * uAspect)
                     + uCamUp    * (sy * uTanHalfFov));
    vec3 c = trace(uCamPos, D);
    c = c / (1.0 + c);                               // Reinhard
    fragColor = vec4(pow(c, vec3(1.0 / 2.2)), 1.0);
}
