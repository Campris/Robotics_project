"""
Scena MuJoCo: Franka Panda (controllo in coppia) + sonda + superficie del torso.

Il torso e' un heightfield con collisioni attive solo contro la sfera della sonda.
La sua quota z = zs(x, y, t) viene aggiornata prima di ogni passo fisico.
La reazione normale e' calcolata dal solver dei contatti MuJoCo, non da Tissue.contact().

Prerequisito: modello Panda dalle Menagerie
    git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git
e imposta PANDA_DIR qui sotto (o la variabile d'ambiente PANDA_DIR).
"""
import os
from pathlib import Path
import numpy as np
import mujoco

PANDA_DIR = Path(os.environ.get("PANDA_DIR", "/home/campri/Scrivania/Robotics/mujoco_menagerie/franka_emika_panda"))

# griglia del torso (heightfield usata sia per collisione sia per visualizzazione)
NC, NR = 81, 61                 # colonne (x), righe (y)
SX, SY = 0.20, 0.15             # semi-dimensioni patch [m]
ELEV, BASE = 0.25, 0.05         # quota massima rappresentabile e spessore sotto lo zero [m]

PANDA_BASE_POS = (-0.55, 0.0, 0.0)   # con -0.40 il Panda (sonda verticale) non raggiunge x < -80 mm; con -0.55 tutta la patch
TIP_Z = 0.107 + 0.15            # punta della sonda lungo l'asse z di link7 [m]
TIP_RADIUS = 0.022              # raggio della sfera terminale collidente [m]
TORQUE_LIMITS = np.array([87, 87, 87, 87, 12, 12, 12], dtype=float)


def build_model():
    spec = mujoco.MjSpec.from_file(str(PANDA_DIR / "panda_nohand.xml"))

    # attuatori: via quelli in posizione, dentro 7 motori in coppia
    for a in list(spec.actuators):
        spec.delete(a)
    for i in range(7):
        a = spec.add_actuator()
        a.name = f"tau{i + 1}"
        a.target = f"joint{i + 1}"
        a.trntype = mujoco.mjtTrn.mjTRN_JOINT
        a.gear[0] = 1.0
        a.ctrllimited = 1
        a.ctrlrange = [-TORQUE_LIMITS[i], TORQUE_LIMITS[i]]

    spec.body("link0").pos = PANDA_BASE_POS

    # Sonda sul link7: cilindro visuale, sfera terminale collidente.
    link7 = spec.body("link7")
    link7.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.018, 0.075, 0],
                   pos=[0, 0, 0.107 + 0.075], mass=0.30, contype=0, conaffinity=0,
                   rgba=[0.15, 0.15, 0.18, 1])
    link7.add_geom(name="tip_vis", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[TIP_RADIUS, 0, 0],
                   pos=[0, 0, TIP_Z - TIP_RADIUS], contype=1, conaffinity=2, rgba=[0.2, 0.8, 0.3, 1])
    link7.add_site(name="tip", pos=[0, 0, TIP_Z], size=[0.003, 0, 0], rgba=[1, 1, 0, 1])

    # Torso: heightfield collidente. La maschera lo abbina solo alla sfera della sonda.
    hf = spec.add_hfield(name="tissue_collision", nrow=NR, ncol=NC, size=[SX, SY, ELEV, BASE])
    hf.userdata = np.full(NR * NC, 0.4)
    visual_hf = spec.add_hfield(name="tissue_visual", nrow=NR, ncol=NC, size=[SX, SY, ELEV, BASE])
    visual_hf.userdata = np.full(NR * NC, 0.4)
    spec.worldbody.add_geom(name="tissue_geom", type=mujoco.mjtGeom.mjGEOM_HFIELD,
                            hfieldname="tissue_collision", contype=2, conaffinity=0,
                            rgba=[0.80, 0.55, 0.45, 0])
    spec.worldbody.add_geom(name="tissue_visual_geom", type=mujoco.mjtGeom.mjGEOM_HFIELD,
                            hfieldname="tissue_visual", contype=0, conaffinity=0,
                            rgba=[0.80, 0.55, 0.45, 1])
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1.0, 1.0, 0.05],
                            contype=0, conaffinity=0, rgba=[0.25, 0.3, 0.35, 1], pos=[0, 0, -BASE - 0.003])
    spec.worldbody.add_light(pos=[0, 0, 1.5], dir=[0, 0, -1],
                            type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                            diffuse=[0.45, 0.45, 0.45], specular=[0, 0, 0])

    model = spec.compile()
    model.opt.timestep = 0.001
    ids = dict(
        site=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tip"),
        tip_geom=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "tip_vis"),
        tissue_geom=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "tissue_geom"),
        hfield=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, "tissue_visual"),
        collision_hfield=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, "tissue_collision"),
    )
    return model, ids


_gx = np.linspace(-SX, SX, NC)
_gy = np.linspace(-SY, SY, NR)
_GX, _GY = np.meshgrid(_gx, _gy)        # shape (NR, NC): righe = y, colonne = x


def update_tissue_mesh(model, tissue, t, vis_gain=1.0, deformation=None):
    """
    Aggiorna il torso fisico e la sua copia grafica. La deformazione da contatto
    viene applicata solo alla copia grafica, senza modificare la collisione MuJoCo.
    deformation: (x, y, penetrazione) nel frame del torso, in metri.
    """
    z0 = tissue.geom.z0(_GX, _GY)
    z = z0 + vis_gain * (tissue.surface_height(_GX, _GY, t) - z0)
    _write_hfield(model, model.hfield("tissue_collision").id, z)

    z_visual = z.copy()
    if deformation is not None:
        x, y, penetration = deformation
        depth = float(np.clip(penetration, 0.0, 0.015))
        if depth > 1e-5:
            radius = np.sqrt(2.0 * TIP_RADIUS * depth)
            sigma = float(np.clip(radius, 0.012, 0.035))
            depression = depth * np.exp(-0.5 * (((_GX - x) / sigma) ** 2 + ((_GY - y) / sigma) ** 2))
            z_visual -= depression
    _write_hfield(model, model.hfield("tissue_visual").id, z_visual)


def measure_tissue_compression(model, data, ids, tissue, t):
    """Stima la compressione geometrica sonda-superficie usando centro e raggio della sfera."""
    center = data.geom_xpos[ids["tip_geom"]]
    x, y = float(center[0]), float(center[1])
    surface = np.array([x, y, tissue.surface_height(x, y, t)])
    normal = tissue.normal(x, y, t)
    radius = float(model.geom_size[ids["tip_geom"], 0])
    signed_distance = float(np.dot(center - surface, normal))
    return max(0.0, radius - signed_distance)


def _write_hfield(model, hfield_id, z):
    start = int(model.hfield_adr[hfield_id])
    count = int(model.hfield_nrow[hfield_id] * model.hfield_ncol[hfield_id])
    model.hfield_data[start:start + count] = np.clip(z / ELEV, 0.0, 1.0).ravel()
