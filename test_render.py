import moderngl, numpy as np, math
from PIL import Image

VERT = """#version 300 es
layout(location=0) in vec2 p;
void main(){ gl_Position = vec4(p, 0.0, 1.0); }
"""
FRAG = open('bh.frag').read()

ctx = moderngl.create_context(standalone=True, backend='egl')
prog = ctx.program(vertex_shader=VERT, fragment_shader=FRAG)
print("shader compiled OK on", ctx.info['GL_RENDERER'])

vbo = ctx.buffer(np.array([-1,-1, 3,-1, -1,3], dtype='f4').tobytes())
vao = ctx.vertex_array(prog, [(vbo, '2f', 'p')])

def camera(rcam, incl_deg, fov_deg, W, H):
    th = math.radians(incl_deg)
    pos = np.array([rcam*math.sin(th), 0.0, rcam*math.cos(th)])
    fwd = -pos/np.linalg.norm(pos)
    wup = np.array([0.0,0.0,1.0])
    right = np.cross(fwd, wup); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return pos, fwd, right, up, math.tan(math.radians(fov_deg)/2), W/H

def render(name, W, H, rcam=30.0, incl=85.0, fov=45.0, rin=6.0, rout=20.0,
           expo=0.65, t=0.0, spin=1.0, steps=768, gr=1):
    pos, fwd, right, up, thf, asp = camera(rcam, incl, fov, W, H)
    fbo = ctx.framebuffer(color_attachments=[ctx.texture((W,H),4)])
    fbo.use(); ctx.viewport = (0,0,W,H)
    prog['uRes'].value = (W,H)
    prog['uCamPos'].value = tuple(pos); prog['uCamFwd'].value = tuple(fwd)
    prog['uCamRight'].value = tuple(right); prog['uCamUp'].value = tuple(up)
    prog['uTanHalfFov'].value = thf; prog['uAspect'].value = asp
    prog['uRcam'].value = rcam
    prog['uRin'].value = rin; prog['uRout'].value = rout
    prog['uExposure'].value = expo; prog['uTime'].value = t
    prog['uSpin'].value = spin; prog['uUEscape'].value = 1.0/max(4*rcam, 400.0)
    prog['uMaxSteps'].value = steps; prog['uGRShade'].value = gr
    vao.render()
    data = np.frombuffer(fbo.read(components=3), dtype=np.uint8).reshape(H,W,3)
    img = data[::-1]                      # GL origin is bottom-left
    Image.fromarray(img).save(name)
    fbo.release()
    return img

# View 1: default edge-on-ish, the money shot
img = render('web_default.png', 900, 500)

# View 2: shadow-radius measurement, disk pushed out of frame
W,H = 1200, 200
rcam, fov = 30.0, 45.0
img2 = render('web_shadow.png', W, H, rin=1e9, rout=1e9, incl=85.0, fov=fov)
row = img2[H//2]
lum = row.astype(int).sum(axis=1)
cx = W/2
# scan outward from center for first non-black pixel
right_edge = next(x for x in range(int(cx), W) if lum[x] > 8)
left_edge  = next(x for x in range(int(cx), 0, -1) if lum[x] > 8)
r_pix = 0.5*((right_edge - cx) + (cx - left_edge))
bc = 3*math.sqrt(3)
alpha = math.asin(bc*math.sqrt(1-2/rcam)/rcam)
thf = math.tan(math.radians(fov)/2)
asp = W/H
sx_edge = math.tan(alpha)/(thf*asp)
r_pred = sx_edge * W/2
print(f"shadow radius: measured {r_pix:.1f}px, analytic {r_pred:.1f}px, "
      f"rel err {abs(r_pix-r_pred)/r_pred:.2e}")

# View 3: face-on, GR shading off vs on (Doppler ring asymmetry check)
a = render('web_faceon.png', 500, 500, incl=15.0, fov=55.0)
b = render('web_flat.png',   900, 500, gr=0)

# Doppler asymmetry, numbers not vibes
d = render('web_dop.png', 600, 340, gr=1).astype(float)
l = d[:, :300].mean(); r = d[:, 300:].mean()
print(f"mean luminance with GR shading: left {l:.1f}, right {r:.1f}, ratio {l/r:.2f}")
d0 = render('web_dop0.png', 600, 340, gr=0).astype(float)
l0 = d0[:, :300].mean(); r0v = d0[:, 300:].mean()
print(f"mean luminance, shading off:    left {l0:.1f}, right {r0v:.1f}, ratio {l0/r0v:.2f}")

# Animation frames actually differ
f1 = render('t0.png', 300, 200, t=0.0); f2 = render('t1.png', 300, 200, t=5.0)
print("animation frame delta (mean abs):", np.abs(f1.astype(int)-f2.astype(int)).mean())
