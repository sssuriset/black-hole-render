# Schwarzschild Ray Tracer (WebGL2)

Real-time general-relativistic ray tracer in a single HTML file. Every pixel
integrates a null geodesic backward from the camera through the Schwarzschild
metric in a GLSL ES 3.00 fragment shader. Runs on integrated graphics; no
build step, no dependencies. Open black-hole-sim.html in any current browser.

## Physics

Schwarzschild geodesics are planar, so the integration runs in each ray's own
orbital plane using the Binet form, u = 1/r, G = c = M = 1:

    d^2u/dphi^2 = 3u^2 - u

Two state variables per ray instead of eight, adaptive RK4, refined near the
photon sphere. The camera is a static observer; ray directions are corrected
by the lapse sqrt(1-2M/r) into its local orthonormal frame. The disk is a
geometrically thin Keplerian annulus from the ISCO (6M) to 20M, shaded with
the exact circular-orbit redshift

    g = sqrt(1-3M/r) / [ sqrt(1-2M/r_cam) (1 - Omega Lz/E) ]

and bolometric beaming g^4 from the invariance of I_nu/nu^3. Disk texture is
advected at the local Keplerian angular rate, so the pattern shears
differentially. The "relativity" toggle sets g = 1 to show what the same disk
looks like without GR shading.

## Verification

The shader was compiled and rendered headlessly under Mesa llvmpipe and
checked quantitatively (test_render.py reproduces this):

- shadow angular radius vs asin(b_c sqrt(1-2M/r)/r): 41.5 px measured vs
  41.0 px predicted at 1200 px wide (edge quantized to whole pixels)
- Doppler asymmetry: left/right luminance ratio 1.43 with GR shading,
  1.00 with the toggle off
- the underlying integrator was validated separately against closed-form
  results: deflection vs the 3rd-order series (rel err 8.8e-5 at b = 100M),
  critical impact parameter vs 3*sqrt(3)*M (rel err 1.0e-3), conserved
  1/b^2 drift 8e-7 over a 611-step near-critical ray

## Controls

Drag to orbit, scroll to dolly (4.6M to 150M), arrows and +/- on the
keyboard. Sliders: field of view, exposure, integrator step budget, render
scale. Buttons: relativity toggle, disk spin direction, pause, reset,
PNG capture. The header readout shows camera radius, inclination, azimuth,
the analytic shadow diameter for the current radius, and live FPS and
throughput.
