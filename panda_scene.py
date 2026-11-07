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
FLANGE_Z = 0.107                # flangia del Panda lungo l'asse z di link7: qui e' montato il sensore F/T [m]
TIP_Z = FLANGE_Z + 0.15         # punta della sonda lungo l'asse z di link7 [m]
PROBE_MASS = 0.30               # massa della sonda (porta-sonda incluso) [kg]
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

    # Sonda: corpo SEPARATO fissato a link7 (nessun giunto), cosi' fra i due c'e' un'interfaccia
    # su cui MuJoCo puo' misurare forza e coppia (il sensore ATI fra flangia e sonda).
    # Nota: link7 nel modello Menagerie ha un <inertial> esplicito, quindi geometrie con massa aggiunte
    # direttamente a link7 verrebbero ignorate. Nel corpo "probe" la massa della sonda conta davvero.
    link7 = spec.body("link7")
    probe = link7.add_body(name="probe")                     # stesso frame di link7
    # Massa della sonda (0.30 kg). Invisibile: arriva fino al fondo della sfera e il suo bordo
    # piatto sporgerebbe sotto di essa.
    probe.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.018, 0.075, 0],
                   pos=[0, 0, FLANGE_Z + 0.075], mass=PROBE_MASS, contype=0, conaffinity=0,
                   rgba=[0.15, 0.15, 0.18, 0])
    # Corpo visibile della sonda (senza massa): finisce al centro della sfera, cosi' l'emisfero
    # inferiore e' il punto piu' basso e tocca il tessuto.
    probe_end = TIP_Z - TIP_RADIUS
    probe.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.018, (probe_end - FLANGE_Z) / 2, 0],
                   pos=[0, 0, (FLANGE_Z + probe_end) / 2], mass=0.0, contype=0, conaffinity=0,
                   rgba=[0.15, 0.15, 0.18, 1])
    probe.add_geom(name="tip_vis", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[TIP_RADIUS, 0, 0],
                   pos=[0, 0, TIP_Z - TIP_RADIUS], mass=0.0, contype=1, conaffinity=2,
                   rgba=[0.2, 0.8, 0.3, 1])
    probe.add_site(name="tip", pos=[0, 0, TIP_Z], size=[0.003, 0, 0], rgba=[1, 1, 0, 1])
    # Sensore F/T sulla flangia: wrench che link7 scambia con la sonda, nel frame del sito "ft".
    probe.add_site(name="ft", pos=[0, 0, FLANGE_Z], size=[0.004, 0, 0], rgba=[1, 0, 1, 1])
    spec.add_sensor(name="ft_force", type=mujoco.mjtSensor.mjSENS_FORCE,
                    objtype=mujoco.mjtObj.mjOBJ_SITE, objname="ft")
    spec.add_sensor(name="ft_torque", type=mujoco.mjtSensor.mjSENS_TORQUE,
                    objtype=mujoco.mjtObj.mjOBJ_SITE, objname="ft")

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
        probe_body=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "probe"),
        ft_site=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ft"),
        ft_adr=int(model.sensor_adr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "ft_force")]),
    )
    # forza (3) e coppia (3) devono essere contigue in sensordata: ft_adr .. ft_adr+6
    assert model.sensor_adr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "ft_torque")] == ids["ft_adr"] + 3
    return model, ids


def apply_tip_force(model, data, body, force, point):
    """
    Applica la forza del tessuto `force` (mondo) nel punto `point` del corpo `body` tramite xfrc_applied.
    xfrc_applied (forza + coppia al baricentro del corpo, in mondo) e' visto anche dai sensori force/torque
    di MuJoCo, a differenza di qfrc_applied. Dinamicamente equivale a mj_applyFT(force, 0, point, body).
    Restituisce le coppie di giunto equivalenti (J^T F), utili per la stima "alla Franka" nel log.
    """
    data.xfrc_applied[body, :3] = force
    data.xfrc_applied[body, 3:] = np.cross(point - data.xipos[body], force)
    tau = np.zeros(model.nv)
    mujoco.mj_applyFT(model, data, force, np.zeros(3), point, body, tau)
    return tau


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
