"""Blender 3D shots for the Mimicry video (Cycles, CPU).  python scenes.py <scene> <seconds> <outdir>
Rendered at 8 fps and 540x960, then interpolated/upscaled with ffmpeg (see build step)."""
import bpy, math, random, sys
from mathutils import Vector
scene_id, D, OUT = sys.argv[-3], float(sys.argv[-2]), sys.argv[-1]
FPS = 6
ORANGE, BLUE, CYAN = (1.0, .42, .18), (.18, .45, 1.0), (.4, .85, 1.0)

bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.render.engine = 'CYCLES'; sc.cycles.samples = 12; sc.cycles.use_adaptive_sampling = True; sc.cycles.max_bounces = 4; sc.cycles.glossy_bounces = 2; sc.cycles.diffuse_bounces = 2; sc.cycles.use_denoising = False
sc.render.use_persistent_data = True
sc.render.resolution_x, sc.render.resolution_y = 450, 800
sc.render.fps = FPS; sc.frame_start = 1; sc.frame_end = int(D * FPS) + 1
sc.view_settings.view_transform = 'AgX'; sc.view_settings.look = 'AgX - Punchy'
sc.render.filepath = OUT + "/f_"
sc.world = bpy.data.worlds.new("w"); sc.world.use_nodes = True
sc.world.node_tree.nodes["Background"].inputs[0].default_value = (0.004, 0.005, 0.009, 1)
random.seed(3)

def mat(name, color, rough=.4, metal=0., emit=0., ecol=None):
    m = bpy.data.materials.new(name); m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*color, 1); b.inputs["Roughness"].default_value = rough; b.inputs["Metallic"].default_value = metal
    b.inputs["Emission Color"].default_value = (*(ecol or color), 1); b.inputs["Emission Strength"].default_value = emit
    return m

def cube(loc, scale, m, bevel=.04):
    bpy.ops.mesh.primitive_cube_add(location=loc); o = bpy.context.object; o.scale = scale
    if bevel:
        mod = o.modifiers.new("b", "BEVEL"); mod.width = bevel; mod.segments = 3
    o.data.materials.append(m); return o

def camera(loc, target, lens=40):
    bpy.ops.object.camera_add(location=loc); c = bpy.context.object; c.data.lens = lens
    bpy.ops.object.empty_add(location=target); t = bpy.context.object
    con = c.constraints.new("TRACK_TO"); con.target = t; con.track_axis = 'TRACK_NEGATIVE_Z'; con.up_axis = 'UP_Y'
    sc.camera = c; return c, t

def light(kind, loc, energy, color=(1, 1, 1), size=2, rot=(0, 0, 0)):
    bpy.ops.object.light_add(type=kind, location=loc, rotation=rot); l = bpy.context.object
    l.data.energy = energy; l.data.color = color
    if kind == 'AREA': l.data.size = size
    return l

def key(obj, attr, frames_vals):
    for f, v in frames_vals:
        setattr(obj, attr, v); obj.keyframe_insert(attr, frame=f)

F = sc.frame_end
ease = lambda x: 1 - (1 - max(0, min(1, x))) ** 3

if scene_id == "potato":   # hook: a real potato floating in an endless tunnel of light frames
    bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32, radius=1); p = bpy.context.object
    p.scale = (.8, .58, .52); bpy.ops.object.shade_smooth()
    tex = bpy.data.textures.new("n", "CLOUDS"); tex.noise_scale = .55
    d = p.modifiers.new("d", "DISPLACE"); d.texture = tex; d.strength = .10
    m = mat("potato", (.42, .28, .14), rough=.62)
    nt = m.node_tree; b = nt.nodes["Principled BSDF"]
    noise = nt.nodes.new("ShaderNodeTexNoise"); noise.inputs["Scale"].default_value = 18
    bump = nt.nodes.new("ShaderNodeBump"); bump.inputs["Strength"].default_value = .35
    nt.links.new(noise.outputs["Fac"], bump.inputs["Height"]); nt.links.new(bump.outputs["Normal"], b.inputs["Normal"])
    spots = nt.nodes.new("ShaderNodeTexVoronoi"); spots.inputs["Scale"].default_value = 7
    ramp = nt.nodes.new("ShaderNodeValToRGB"); ramp.color_ramp.elements[0].position = .06
    ramp.color_ramp.elements[0].color = (.12, .07, .03, 1); ramp.color_ramp.elements[1].color = (.46, .31, .15, 1)
    nt.links.new(spots.outputs["Distance"], ramp.inputs["Fac"]); nt.links.new(ramp.outputs["Color"], b.inputs["Base Color"])
    p.data.materials.append(m)
    key(p, "rotation_euler", [(1, (.2, 0, 0)), (F, (.5, .3, math.radians(140)))])
    key(p, "location", [(1, (0, 0, -.05)), (F // 2, (0, 0, .12)), (F, (0, 0, 0))])
    mo, mb = mat("fo", ORANGE, emit=2.5), mat("fb", BLUE, emit=3)
    for i in range(18):   # receding frames = the "infinite mirror"
        y = 2.2 + i * 2.2; mm = mo if i % 2 == 0 else mb
        for loc, s in [((0, y, 2.2), (2.6, .04, .04)), ((0, y, -2.2), (2.6, .04, .04)), ((-2.6, y, 0), (.04, .04, 2.2)), ((2.6, y, 0), (.04, .04, 2.2))]:
            cube(loc, s, mm, bevel=0)
    floor = cube((0, 20, -2.4), (6, 40, .05), mat("floor", (.02, .02, .025), rough=.12, metal=.6), bevel=0)
    light('SPOT', (0, -1.5, 4.5), 900, (1, .92, .8)).data.spot_size = .7
    light('AREA', (-3, -3, 1), 120, (.6, .7, 1), size=3)
    c, t = camera((.3, -7.8, .3), (0, 3, 0), lens=42)
    key(c, "location", [(1, (.3, -7.8, .3)), (F, (-.1, -5.4, .12))])

elif scene_id == "bars":   # trained in: base vs assistant claim rates as 3D blocks
    vals = [("0.5B base", .33, ORANGE), ("0.5B asst", 0.0, BLUE), ("1.5B base", .67, ORANGE), ("1.5B asst", .50, BLUE)]
    cube((0, 0, -.06), (8, 8, .05), mat("floor", (.015, .016, .02), rough=.2, metal=.3), bevel=0)
    grid = mat("grid", (.1, .12, .16), emit=.6, ecol=(.2, .3, .5))
    for i in range(-7, 8):
        cube((i * .8, 0, 0), (.006, 8, .006), grid, bevel=0); cube((0, i * .8, 0), (8, .006, .006), grid, bevel=0)
    for i, (name, v, col) in enumerate(vals):
        h = max(v * 3.6, .05); x = -2.1 + i * 1.4
        o = cube((x, 0, h / 2), (.5, .5, h / 2), mat(name, col, rough=.25, emit=.8), bevel=.05)
        o.scale.z = .001; o.location.z = 0
        for f in range(1, F + 1):
            k = ease((f / FPS - .3 - i * .35) / 1.4); o.scale.z = max(.001, h / 2 * k); o.location.z = h / 2 * k
            o.keyframe_insert("scale", frame=f); o.keyframe_insert("location", frame=f)
    light('AREA', (3, -4, 6), 700, (1, .95, .9), size=5); light('AREA', (-5, 2, 3), 250, (.5, .6, 1), size=4)
    c, t = camera((6, -7, 3.2), (0, 0, .9), lens=40)
    for f in range(1, F + 1):
        a = -.9 + .35 * f / F; c.location = (11.5 * math.sin(a) * -1, -11.5 * math.cos(a), 4.2 - .6 * f / F); c.keyframe_insert("location", frame=f)
    c.data.shift_y = .18

elif scene_id == "grid":   # inside the model: a pulse spreading through a lattice of activations
    NX, NZ = 11, 15; objs = []
    for ix in range(NX):
        for iz in range(NZ):
            m = mat(f"c{ix}_{iz}", (.03, .05, .09), rough=.35, emit=.0, ecol=CYAN)
            o = cube(((ix - NX / 2) * .42, random.uniform(-.08, .08), (iz - NZ / 2) * .42), (.16, .16, .16), m, bevel=.03)
            objs.append((o, m, math.hypot(ix - 5, iz - 9)))
    for o, m, d in objs:
        s = m.node_tree.nodes["Principled BSDF"].inputs["Emission Strength"]
        for f in range(1, F + 1):
            t = f / FPS; front = max(0, (t - 1.2) * 1.1)
            s.default_value = 5 * math.exp(-((d - front) ** 2) / 1.0) + (.25 if d < front else 0)
            s.keyframe_insert("default_value", frame=f)
    light('AREA', (0, -6, 4), 300, (.8, .85, 1), size=6)
    c, t = camera((-2.5, -8, 1.5), (0, 0, -.8), lens=42)
    for f in range(1, F + 1):
        a = -.3 + .5 * f / F; c.location = (10.5 * math.sin(a), -10.5 * math.cos(a), 1.4); c.keyframe_insert("location", frame=f)
    c.data.shift_y = .15

elif scene_id == "mirror":   # the verdict: a glowing mind in front of a mirror that can't show it clearly
    g = bpy.data.collections.new("pts")
    mo = mat("pt", ORANGE, emit=2.2); n = 260
    parent = bpy.data.objects.new("P", None); sc.collection.objects.link(parent)
    for i in range(n):
        y = 1 - 2 * (i + .5) / n; r = math.sqrt(1 - y * y); th = math.pi * (3 - math.sqrt(5)) * i
        o = cube((math.cos(th) * r * 1.1, math.sin(th) * r * 1.1, y * 1.1), (.035, .035, .035), mo, bevel=0); o.parent = parent
    parent.location = (0, 0, 0)
    key(parent, "rotation_euler", [(1, (0, 0, 0)), (F, (0, .3, math.radians(80)))])
    cube((0, 2.6, 0), (2.4, .05, 3.2), mat("mirror", (.9, .9, .92), rough=.32, metal=1.0), bevel=.02)   # frosted mirror
    cube((0, 2.62, 0), (2.55, .04, 3.35), mat("frame", (.05, .04, .03), rough=.4, metal=.8), bevel=.03)
    cube((0, 0, -3.3), (8, 8, .05), mat("floor", (.02, .02, .025), rough=.15, metal=.5), bevel=0)
    light('AREA', (-3, -3, 3), 250, (.6, .7, 1), size=4)
    c, t = camera((2.8, -6.2, .4), (0, 1.2, -.2), lens=36)
    key(c, "location", [(1, (3.0, -6.4, .5)), (F, (1.6, -5.0, .2))])
    c.data.shift_y = .12

bpy.ops.render.render(animation=True)
print("DONE", scene_id, F, "frames")
