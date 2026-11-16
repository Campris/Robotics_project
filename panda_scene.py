"""
Scena MuJoCo: Franka Panda (controllo in coppia) + sonda con sensore F/T + superficie del torso (solo grafica).

Il contatto NON e' calcolato da MuJoCo: la forza del tessuto viene da Tissue.contact() (Hunt-Crossley) e
viene applicata alla sonda con apply_tip_force(). Il torso e' un heightfield senza collisioni, aggiornato
solo per la grafica.

Il modello del Panda (panda_nohand.xml, dalle MuJoCo Menagerie di Google DeepMind) e' nella cartella
franka_emika_panda/ di questo repository. Un altro percorso si puo' indicare con la variabile d'ambiente PANDA_DIR.
"""

import os
from pathlib import Path
import numpy as np
import mujoco

PANDA_DIR = Path(os.environ.get("PANDA_DIR", Path(__file__).resolve().parent / "franka_emika_panda"))

NC, NR = 81, 61                 # punti della griglia del torso: colonne (x), righe (y)
SX, SY = 0.20, 0.15             # semi-dimensioni patch [m]
ELEV, BASE = 0.25, 0.05         # quota massima rappresentabile e spessore sotto lo zero [m]

PANDA_BASE_POS = (-0.55, 0.0, 0.0)   # posizione della base del robot nel mondo [m]
FLANGE_Z = 0.107                # flangia del Panda lungo l'asse z di link7: qui e' montato il sensore F/T [m]
TIP_Z = FLANGE_Z + 0.15         # punta della sonda lungo l'asse z di link7 [m]
PROBE_MASS = 0.30               # massa della sonda (porta-sonda incluso) [kg]
TIP_RADIUS = 0.022              # raggio della sfera terminale [m]
TORQUE_LIMITS = np.array([87, 87, 87, 87, 12, 12, 12], dtype=float)
Q_HOVER = (-0.2060103872267039, -0.19019021558822838, -0.07189872951261951, -2.6899356513525126,
           -0.022682449347206736, 2.5000783113337017, -0.25844787553135584)


def build_model():
    spec = mujoco.MjSpec.from_file(str(PANDA_DIR / "panda_nohand.xml"))

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

    link7 = spec.body("link7")
    probe = link7.add_body(name="probe")
    probe.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.018, 0.075, 0],
                   pos=[0, 0, FLANGE_Z + 0.075], mass=PROBE_MASS, contype=0, conaffinity=0,
                   rgba=[0.15, 0.15, 0.18, 0])
    probe_end = TIP_Z - TIP_RADIUS
    probe.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.018, (probe_end - FLANGE_Z) / 2, 0],
                   pos=[0, 0, (FLANGE_Z + probe_end) / 2], mass=0.0, contype=0, conaffinity=0,
                   rgba=[0.15, 0.15, 0.18, 1])
    probe.add_geom(name="tip_vis", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[TIP_RADIUS, 0, 0],
                   pos=[0, 0, TIP_Z - TIP_RADIUS], mass=0.0, contype=0, conaffinity=0,
                   rgba=[0.2, 0.8, 0.3, 1])
    probe.add_site(name="tip", pos=[0, 0, TIP_Z], size=[0.003, 0, 0], rgba=[1, 1, 0, 1])
    probe.add_site(name="ft", pos=[0, 0, FLANGE_Z], size=[0.004, 0, 0], rgba=[1, 0, 1, 1])
    spec.add_sensor(name="ft_force", type=mujoco.mjtSensor.mjSENS_FORCE,
                    objtype=mujoco.mjtObj.mjOBJ_SITE, objname="ft")
    spec.add_sensor(name="ft_torque", type=mujoco.mjtSensor.mjSENS_TORQUE,
                    objtype=mujoco.mjtObj.mjOBJ_SITE, objname="ft")

    hf = spec.add_hfield(name="tissue_visual", nrow=NR, ncol=NC, size=[SX, SY, ELEV, BASE])
    hf.userdata = np.full(NR * NC, 0.4)
    spec.worldbody.add_geom(name="tissue_visual_geom", type=mujoco.mjtGeom.mjGEOM_HFIELD,
                            hfieldname="tissue_visual", contype=0, conaffinity=0,
                            rgba=[0.80, 0.55, 0.45, 1])
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1.0, 1.0, 0.05],
                            contype=0, conaffinity=0, rgba=[0.25, 0.3, 0.35, 1], pos=[0, 0, -BASE - 0.003])
    spec.worldbody.add_light(pos=[0, 0, 1.5], dir=[0, 0, -1],
                             type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                             diffuse=[0.45, 0.45, 0.45], specular=[0, 0, 0])

    spec.add_key(name="hover", qpos=Q_HOVER)

    model = spec.compile()
    model.opt.timestep = 0.001
    ids = dict(
        site=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tip"),
        tip_geom=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "tip_vis"),
        hfield=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, "tissue_visual"),
        probe_body=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "probe"),
        ft_site=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ft"),
        ft_adr=int(model.sensor_adr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "ft_force")]),
    )
    assert model.sensor_adr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "ft_torque")] == ids["ft_adr"] + 3
    return model, ids


def rot_err(R, Rd):
    """Errore di orientamento (vettore 3D, piccolo angolo) che porta R verso Rd."""
    return 0.5 * (np.cross(R[:, 0], Rd[:, 0]) + np.cross(R[:, 1], Rd[:, 1]) + np.cross(R[:, 2], Rd[:, 2]))


def jac(model, data, site):
    """Jacobiano 6 x nv del sito: righe 0-2 velocita' lineare, righe 3-5 velocita' angolare."""
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jp, jr, site)
    return np.vstack([jp, jr])


def apply_tip_force(data, body, force, point):
    """
    Applica la forza del tessuto `force` (mondo) nel punto `point` del corpo `body` tramite xfrc_applied:
    forza + coppia rispetto al baricentro del corpo, in mondo. A differenza di qfrc_applied, xfrc_applied
    e' visto anche dai sensori force/torque di MuJoCo (il sensore ATI alla flangia).
    """
    data.xfrc_applied[body, :3] = force
    data.xfrc_applied[body, 3:] = np.cross(point - data.xipos[body], force)


_gx = np.linspace(-SX, SX, NC)
_gy = np.linspace(-SY, SY, NR)
_GX, _GY = np.meshgrid(_gx, _gy)


def update_tissue_mesh(model, tissue, t, vis_gain=1.0, deformation=None):
    """
    Aggiorna la grafica del torso: superficie al tempo t (respiro esagerato di vis_gain) e,
    se deformation = (x, y, penetrazione) [m], l'avvallamento sotto la sonda.
    """
    z0 = tissue.geom.z0(_GX, _GY)
    z = z0 + vis_gain * (tissue.surface_height(_GX, _GY, t) - z0)
    if deformation is not None:
        x, y, penetration = deformation
        depth = float(np.clip(penetration, 0.0, 0.015))
        if depth > 1e-5:
            radius = np.sqrt(2.0 * TIP_RADIUS * depth)
            sigma = float(np.clip(radius, 0.012, 0.035))
            z = z - depth * np.exp(-0.5 * (((_GX - x) / sigma) ** 2 + ((_GY - y) / sigma) ** 2))
    hid = model.hfield("tissue_visual").id
    start = int(model.hfield_adr[hid])
    model.hfield_data[start:start + NR * NC] = np.clip(z / ELEV, 0.0, 1.0).ravel()


def measure_tissue_compression(model, data, ids, tissue, t):
    """Compressione geometrica sfera-superficie (centro e raggio della sfera): serve solo alla grafica."""
    center = data.geom_xpos[ids["tip_geom"]]
    x, y = float(center[0]), float(center[1])
    surface = np.array([x, y, tissue.surface_height(x, y, t)])
    normal = tissue.normal(x, y, t)
    radius = float(model.geom_size[ids["tip_geom"], 0])
    signed_distance = float(np.dot(center - surface, normal))
    return max(0.0, radius - signed_distance)
