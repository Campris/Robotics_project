"""
Demo: Panda (coppia) con sonda che scende sul torso che respira e misura la forza normale.

  python demo_panda.py                 # viewer interattivo (finestra MuJoCo)
  python demo_panda.py --headless      # nessuna finestra: grafico + immagini in ./out

La sonda tiene una POSIZIONE (impedenza cartesiana).

Due modelli di contatto (--contact-model):
  hc      (default) la forza sulla punta e' quella di Tissue.contact() (Hunt-Crossley, rigidita' k_eq(x,y),
          velocita' della pelle inclusa) e viene applicata con mj_applyFT. Heightfield e sfera sono solo grafica.
  native  contatto nativo MuJoCo sonda-heightfield. La forza e' la somma VETTORIALE dei contatti (vedi
          contact_force) e viene validata con la stima da coppie di giunto (fext_from_torques).
          Attenzione: la rigidita' effettiva dipende dal numero di contatti dei prismi, non da k_eq.

Colore della punta: verde = forza nel range 8-10 N, giallo = fuori range, rosso = > 15 N.
"""
import argparse
import time
from pathlib import Path
import numpy as np
import mujoco

from tissue import Tissue, Event, ForceSensor, random_events
import panda_scene as ps

# ---------------------------------------------------------------- parametri da giocarci
X0, Y0 = 0.0, 0.0              # punto di contatto iniziale sul tessuto [m]
HOVER = 0.06                   # altezza di partenza sopra la superficie a riposo [m]
DEPTH = 0.008                  # profondita' target sotto la superficie a riposo [m]
T_DESCENT = 8.0                # durata della discesa [s]
# Esperimenti: prova per esempio 4, 8, 12 s. A parita' di percorso,
# la velocita' massima della traiettoria min-jerk e' circa 1.875*(HOVER+DEPTH)/T_DESCENT.
SLIDE = None                   # es. (-0.08, 0.05): scorre da x=-0.08 a x=0.05 dopo la discesa
F_DES, F_TOL = 9.0, 1.0        # range "bersaglio" solo per il colore e le statistiche
CONTACT_TIME_SCALE = 40.0      # fissato per il confronto: non modificarlo negli esperimenti velocita'/rigidita'
# Scale globali/locali della mappa k_eq [N/m]. Global=1 e tutte le scale zona=1
# riproducono i valori originali; prova una variazione alla volta (es. 0.5, 1, 2).
STIFFNESS_SCALE = 1.0
K_ABDOMEN_SCALE = 1.0
K_CHEST_SCALE = 1.0
K_RIB_SCALE = 1.0
K_STERNUM_SCALE = 1.0
# Esempi: --compare-contact-models --duration 10 --descent-time 12
#         --compare-contact-models --duration 10 --stiffness-scale 0.5
#         --compare-contact-models --duration 10 --k-abdomen-scale 2
# Il fattore CONTACT_TIME_SCALE resta intenzionalmente fisso a 40 in queste prove.

KP = np.array([800.0, 800.0, 600.0, 60.0, 60.0, 60.0])   # [N/m x3, Nm/rad x3]
ZETA = 1.0                                               # smorzamento critico per asse
K_NULL, D_NULL = 5.0, 2.0                                # postura nel nullspace
# --------------------------------------------------------------------------------------


def rot_err(R, Rd):
    return 0.5 * (np.cross(R[:, 0], Rd[:, 0]) + np.cross(R[:, 1], Rd[:, 1]) + np.cross(R[:, 2], Rd[:, 2]))


def jac(model, data, site):
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jp, jr, site)
    return np.vstack([jp, jr])


def ik(model, data, site, p_t, R_t, q_home, iters=400):
    for _ in range(iters):
        mujoco.mj_forward(model, data)
        e = np.concatenate([p_t - data.site_xpos[site],
                            rot_err(data.site_xmat[site].reshape(3, 3), R_t)])
        if np.linalg.norm(e) < 1e-5:
            break
        J = jac(model, data, site)
        Jp = J.T @ np.linalg.inv(J @ J.T + 1e-4 * np.eye(6))
        dq = Jp @ e + (np.eye(model.nv) - Jp @ J) @ (0.1 * (q_home - data.qpos[:model.nv]))
        data.qpos[:model.nv] += 0.5 * dq
    mujoco.mj_forward(model, data)
    return np.linalg.norm(e)


class Sim:
    def __init__(self, seed=0, random_ev=False, solver=None, cone=None,
                 descent_time=T_DESCENT, stiffness_scale=STIFFNESS_SCALE,
                 regional_stiffness_scales=None, contact_model="hc"):
        if descent_time <= 0 or stiffness_scale <= 0:
            raise ValueError("Durata discesa e scala stiffness devono essere positive")
        if contact_model not in ("hc", "native"):
            raise ValueError("contact_model deve essere 'hc' o 'native'")
        self.contact_model = contact_model
        self.model, self.ids = ps.build_model()
        if contact_model == "hc":
            # niente contatto nativo: la forza viene da Tissue.contact() e applicata alla punta
            for g in (self.ids["tissue_geom"], self.ids["tip_geom"]):
                self.model.geom_contype[g] = 0
                self.model.geom_conaffinity[g] = 0
            self.body = int(self.model.site_bodyid[self.ids["site"]])
        if solver is not None:
            self.model.opt.solver = solver
        if cone is not None:
            self.model.opt.cone = cone
        self.contact_time_scale = CONTACT_TIME_SCALE
        self.descent_time = float(descent_time)
        self.stiffness_scale = float(stiffness_scale)
        self.data = mujoco.MjData(self.model)
        m, d, ids = self.model, self.data, self.ids
        self.dt = m.opt.timestep

        events = [Event(16.0, 0.4, 0.005, "bump"), Event(22.0, 1.0, 0.004, "step")]
        if random_ev:
            events = random_events(np.random.default_rng(seed), 120.0)
        self.tissue = Tissue(events=events)
        scales = regional_stiffness_scales or {
            "k_abdomen": K_ABDOMEN_SCALE,
            "k_chest": K_CHEST_SCALE,
            "k_rib": K_RIB_SCALE,
            "k_sternum": K_STERNUM_SCALE,
        }
        for region, scale in scales.items():
            if region not in ("k_abdomen", "k_chest", "k_rib", "k_sternum"):
                raise ValueError(f"Regione stiffness sconosciuta: {region}")
            if scale <= 0:
                raise ValueError(f"La scala di {region} deve essere positiva")
            setattr(self.tissue.stiff, region, getattr(self.tissue.stiff, region) * scale)
        for region in ("k_abdomen", "k_chest", "k_rib", "k_sternum"):
            setattr(
                self.tissue.stiff, region,
                getattr(self.tissue.stiff, region) * self.stiffness_scale,
            )
        self.sensor = ForceSensor(self.dt, seed=seed)
        self.native_plot_filter = ForceSensor(self.dt, noise_std=0.0, delay_s=0.003, seed=seed)
        self.hc_plot_filter = ForceSensor(self.dt, noise_std=0.0, delay_s=0.003, seed=seed)

        mujoco.mj_resetDataKeyframe(m, d, 0)
        q_home = d.qpos[:m.nv].copy()
        self.R_des = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1.0]])      # asse z della sonda verso il basso
        self.z_rest = float(self.tissue.geom.z0(X0, Y0))
        err = ik(m, d, ids["site"], np.array([X0, Y0, self.z_rest + HOVER]), self.R_des, q_home)
        print(f"IK errore residuo: {err:.2e}")
        self.q_null = d.qpos[:m.nv].copy()
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        self.log = {k: [] for k in (
            "t", "fn", "ft", "fresultant", "fmeas", "fn_lpf", "fhc", "fhc_lpf",
            "delta", "ddelta", "k", "solref_tc", "ncontact",
            "zs", "ztip", "x", "tau", "fext_tau", "fcmd_z", "compression",
        )}
        self.visual_deformation = None          # (x, y, compressione) per la grafica del tessuto

    # riferimento cartesiano (qui entrera' l'ammettenza)
    def target(self, t):
        s = np.clip(t / self.descent_time, 0, 1)
        mj = 10 * s**3 - 15 * s**4 + 6 * s**5
        x = X0
        if SLIDE is not None:
            u = np.clip((t - self.descent_time - 4.0) / 10.0, 0, 1)
            x = SLIDE[0] + (SLIDE[1] - SLIDE[0]) * (10 * u**3 - 15 * u**4 + 6 * u**5) if t > self.descent_time else SLIDE[0]
        z0 = self.z_rest + HOVER
        z1 = self.z_rest - DEPTH
        return np.array([x, Y0, z0 + (z1 - z0) * mj])

    def configure_contact_stiffness(self):
        """Imposta la risposta del materiale in base alla zona sotto la sonda."""
        m, d = self.model, self.data
        probe_geom = self.ids["tip_geom"]
        tissue_geom = self.ids["tissue_geom"]
        probe_body = int(m.geom_bodyid[probe_geom])
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        point = d.site_xpos[self.ids["site"]].copy()
        normal = self.tissue.normal(point[0], point[1], d.time)
        k_eq = float(self.tissue.stiff(point[0], point[1]))

        Minv = np.zeros((m.nv, m.nv))
        mujoco.mj_solveM(m, d, Minv, np.eye(m.nv))
        contact_point = point - ps.TIP_RADIUS * normal
        mujoco.mj_jac(m, d, jacp, jacr, contact_point, probe_body)
        jac_normal = normal @ jacp
        inverse_effective_mass = float(jac_normal @ Minv @ jac_normal)
        if not np.isfinite(inverse_effective_mass) or inverse_effective_mass <= 0.0:
            raise RuntimeError("Massa efficace non valida sotto la sonda")

        effective_mass = 1.0 / inverse_effective_mass
        # solref positive uses (time constant, damping ratio); match its
        # characteristic time to the local mass-spring timescale.
        timeconst = max(
            self.contact_time_scale * np.sqrt(effective_mass / k_eq),
            2.0 * self.dt,
        )
        m.geom_solref[probe_geom] = (timeconst, 1.0)
        m.geom_solref[tissue_geom] = (timeconst, 1.0)
        return k_eq, timeconst

    def contact_force(self):
        """
        Forza esterna sulla punta dai contatti sonda-tessuto di MuJoCo.

        Ogni contatto da' una forza nel proprio frame (normale, tang1, tang2). Con l'heightfield i contatti
        sono molti (prismi) e con normali diverse: i moduli NON si sommano, si sommano i VETTORI nel mondo.
        La forza normale e' poi la componente lungo la normale della pelle sotto la punta.
        Ritorna: (fn, ft, |F|, penetrazione, k_eq medio, solref medio, n. contatti).
        """
        m, d = self.model, self.data
        tip, tis = self.ids["tip_geom"], self.ids["tissue_geom"]
        f_world = np.zeros(3)                  # forza SULLA punta [N], frame mondo
        wrench = np.zeros(6)
        penetration = 0.0
        stiffness_sum = 0.0
        timeconst_sum = 0.0
        contact_count = 0
        rotation = d.geom_xmat[tis].reshape(3, 3)
        for i in range(d.ncon):
            c = d.contact[i]
            if {c.geom1, c.geom2} != {tip, tis}:
                continue
            mujoco.mj_contactForce(m, d, i, wrench)
            f_i = c.frame.reshape(3, 3).T @ wrench[:3]          # frame -> mondo
            f_world += f_i if c.geom2 == tip else -f_i          # la normale va da geom1 a geom2
            penetration = max(penetration, max(0.0, -c.dist))
            local_pos = rotation.T @ (c.pos - d.geom_xpos[tis])
            stiffness_sum += float(self.tissue.stiff(local_pos[0], local_pos[1]))
            timeconst_sum += float(c.solref[0])
            contact_count += 1
        self.last_contact_force = f_world
        if contact_count == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
        p = d.site_xpos[self.ids["site"]]
        n = self.tissue.normal(p[0], p[1], d.time)             # normale della pelle sotto la punta
        fn = max(0.0, float(f_world @ n))
        ft = float(np.linalg.norm(f_world - (f_world @ n) * n))
        return (fn, ft, float(np.linalg.norm(f_world)), penetration,
                stiffness_sum / contact_count, timeconst_sum / contact_count, contact_count)

    def fext_from_torques(self):
        """
        Stima della forza esterna sulla punta dalle coppie di giunto, come fa il Panda per F_ext:
        J^T F = tau_ext, con tau_ext = qfrc_constraint (qui solo contatti). Utile per validare contact_force().
        """
        nv = self.model.nv
        return np.linalg.lstsq(self._J_pos.T, self.data.qfrc_constraint[:nv], rcond=None)[0]

    def hunt_crossley_force(self):
        """Stima Hunt-Crossley sul manifold corrente; la forza non viene applicata al Panda."""
        m, d = self.model, self.data
        probe_geom = self.ids["tip_geom"]
        tissue_geom = self.ids["tissue_geom"]
        probe_body = int(m.geom_bodyid[probe_geom])
        rotation = d.geom_xmat[tissue_geom].reshape(3, 3)
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        contacts = [
            d.contact[i] for i in range(d.ncon)
            if {d.contact[i].geom1, d.contact[i].geom2} == {probe_geom, tissue_geom}
        ]
        if not contacts:
            return 0.0, 0.0

        # k_eq descrive la rigidezza complessiva: la distribuiamo sui punti
        # del manifold per evitare che la forza cresca solo aumentando ncon.
        point_k_scale = self.tissue.delta_ref ** (1.0 - self.tissue.n) / len(contacts)
        total_force = 0.0
        max_closing_speed = 0.0
        h = 0.5 * self.dt
        for contact in contacts:
            local_pos = rotation.T @ (contact.pos - d.geom_xpos[tissue_geom])
            x, y = float(local_pos[0]), float(local_pos[1])
            k_eq = float(self.tissue.stiff(x, y))
            normal = contact.frame[:3].copy()
            surface_normal = self.tissue.normal(x, y, d.time)
            if np.dot(normal, surface_normal) < 0:
                normal = -normal

            mujoco.mj_jac(m, d, jacp, jacr, contact.pos, probe_body)
            probe_velocity = jacp @ d.qvel[:m.nv]
            surface_velocity = np.array([
                0.0,
                0.0,
                (self.tissue.surface_height(x, y, d.time + h)
                 - self.tissue.surface_height(x, y, d.time - h)) / (2 * h),
            ])
            closing_speed = -float(np.dot(probe_velocity - surface_velocity, normal))
            max_closing_speed = max(max_closing_speed, closing_speed)

            raw_delta = max(0.0, -float(contact.dist))
            delta = self.tissue.eps * np.logaddexp(0.0, raw_delta / self.tissue.eps)
            delta_rate = float(1.0 / (1.0 + np.exp(-np.clip(raw_delta / self.tissue.eps, -60, 60)))) * closing_speed
            k = (k_eq * point_k_scale)
            damping = max(0.0, 1.0 + 1.5 * self.tissue.alpha * delta_rate)
            total_force += k * delta**self.tissue.n * damping
        return total_force, max_closing_speed

    def step(self):
        m, d, ids = self.model, self.data, self.ids
        t = d.time
        native = self.contact_model == "native"
        if native:
            # La geometria usata dal solver viene aggiornata prima del rilevamento contatti.
            ps.update_tissue_mesh(m, self.tissue, t)
            k_candidate, tc_candidate = self.configure_contact_stiffness()
        mujoco.mj_step1(m, d)

        compression = ps.measure_tissue_compression(m, d, ids, self.tissue, t)
        p = d.site_xpos[ids["site"]].copy()
        R = d.site_xmat[ids["site"]].reshape(3, 3)
        J = jac(m, d, ids["site"])
        v = J @ d.qvel[:m.nv]

        if not native:
            # forza del tessuto sulla punta (Hunt-Crossley), applicata nel punto di contatto
            hc = self.tissue.contact(p, v[:3], t)
            # xfrc_applied (non qfrc_applied): cosi' anche il sensore F/T della flangia vede la forza
            tau_hc = ps.apply_tip_force(m, d, self.body, hc.force, p)

        # impedenza cartesiana + compensazione di gravita'/Coriolis + postura nel nullspace
        Minv = np.zeros((m.nv, m.nv))
        mujoco.mj_solveM(m, d, Minv, np.eye(m.nv))      # M^-1 (le righe sono le colonne risolte, M simmetrica)
        Lam = np.linalg.inv(J @ Minv @ J.T + 1e-9 * np.eye(6))
        D = 2 * ZETA * np.sqrt(KP * np.diag(Lam))
        e = np.concatenate([self.target(t) - p, rot_err(R, self.R_des)])
        F = KP * e - D * v
        tau = J.T @ F
        Jbar = Minv @ J.T @ Lam
        tau0 = K_NULL * (self.q_null - d.qpos[:m.nv]) - D_NULL * d.qvel[:m.nv]
        tau += (np.eye(m.nv) - J.T @ Jbar.T) @ tau0
        d.ctrl[:] = np.clip(tau + d.qfrc_bias[:m.nv], -ps.TORQUE_LIMITS, ps.TORQUE_LIMITS)
        self._J_pos = J[:3].copy()          # per fext_from_torques()
        self.last_F_cmd = F.copy()          # forza cartesiana comandata dall'impedenza
        mujoco.mj_step2(m, d)

        if native:
            fn, ft, fresultant, penetration, k_eq, timeconst, ncontact = self.contact_force()
            fhc, closing_speed = self.hunt_crossley_force()
            if ncontact == 0:
                k_eq, timeconst = k_candidate, tc_candidate
            tau_ext = d.qfrc_constraint[:m.nv]            # coppie dovute ai contatti
        else:
            fn, ft, fresultant = hc.fn, 0.0, hc.fn
            fhc, closing_speed = hc.fn, hc.ddelta
            penetration, k_eq, timeconst, ncontact = max(hc.delta, 0.0), hc.k_eq, 0.0, int(hc.fn > 1e-3)
            tau_ext = tau_hc                              # coppie della forza applicata (stima "come il Panda")
        L = self.log
        L["t"].append(t); L["fn"].append(fn); L["ft"].append(ft)
        L["fresultant"].append(fresultant); L["fmeas"].append(self.sensor.read(fn))
        L["fn_lpf"].append(self.native_plot_filter.read(fn))
        L["fhc"].append(fhc); L["fhc_lpf"].append(self.hc_plot_filter.read(fhc))
        L["delta"].append(penetration); L["ddelta"].append(closing_speed); L["k"].append(k_eq)
        L["solref_tc"].append(timeconst); L["ncontact"].append(ncontact)
        L["zs"].append(self.tissue.surface_height(p[0], p[1], t)); L["ztip"].append(p[2]); L["x"].append(p[0])
        L["tau"].append(np.abs(d.ctrl[:]).max())
        pn = self.tissue.normal(p[0], p[1], t)
        L["fext_tau"].append(float(np.linalg.lstsq(self._J_pos.T, tau_ext, rcond=None)[0] @ pn))
        L["fcmd_z"].append(float(-self.last_F_cmd[2]))     # spinta verso il basso comandata dall'impedenza
        L["compression"].append(compression)
        centre = d.geom_xpos[ids["tip_geom"]]
        self.visual_deformation = (float(centre[0]), float(centre[1]), compression) if compression > 0 else None
        return fn

    def color_tip(self, f):
        rgba = (0.2, 0.8, 0.3, 1) if abs(f - F_DES) <= F_TOL else ((0.9, 0.2, 0.2, 1) if f > 15 else (0.95, 0.8, 0.2, 1))
        self.model.geom_rgba[self.ids["tip_geom"]] = rgba


def run_viewer(sim, duration, vis_gain=1.0):
    import mujoco.viewer
    m, d = sim.model, sim.data
    with mujoco.viewer.launch_passive(m, d) as v:
        v.cam.azimuth, v.cam.elevation, v.cam.distance = 140, -25, 1.3
        v.cam.lookat[:] = [0.0, 0.0, 0.15]
        next_print = 0.0
        while v.is_running() and d.time < duration:
            t_wall = time.time()
            for _ in range(16):                                   # ~16 ms di simulazione per frame
                fn = sim.step()
            ps.update_tissue_mesh(m, sim.tissue, d.time, vis_gain, sim.visual_deformation)
            sim.color_tip(fn)
            v.update_hfield(sim.ids["hfield"])
            v.sync()
            if d.time >= next_print:
                print(f"t={d.time:6.2f}s  penetrazione {sim.log['delta'][-1]*1000:6.2f} mm  "
                      f"F contatto {fn:5.2f} N  F sensore {sim.log['fmeas'][-1]:5.2f} N  "
                      f"k_eq {sim.log['k'][-1]:5.0f} N/m  contatti {sim.log['ncontact'][-1]:d}  "
                      f"solref tc {sim.log['solref_tc'][-1]*1000:5.1f} ms")
                next_print += 0.5
            time.sleep(max(0.0, 0.016 - (time.time() - t_wall)))


class LiveMonitor:
    def __init__(self, cases, interactive=True):
        import matplotlib
        if interactive and matplotlib.get_backend().lower() == "agg":
            try:
                import tkinter
            except ImportError as exc:
                raise RuntimeError(
                    "Grafici live non disponibili: installa il supporto Tk per Python 3.14 "
                    "(Ubuntu: sudo apt install python3.14-tk), oppure usa --headless per salvare i grafici."
                ) from exc
            matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt

        self.plt = plt
        self.interactive = interactive
        self.fig, self.axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        self.fig.suptitle("Contatto Panda-tessuto: confronto solver MuJoCo")
        self.axes[0].set_ylabel("Forza normale [N]")
        self.axes[1].set_ylabel("Forza tangenziale [N]")
        self.axes[2].set_ylabel("Penetrazione [mm]")
        self.axes[3].set_ylabel("Coppia max [Nm]")
        self.axes[3].set_xlabel("Tempo simulato [s]")
        self.k_axis = self.axes[2].twinx()
        self.k_axis.set_ylabel("Rigidita' locale [N/m]")
        for ax in self.axes:
            ax.grid(alpha=0.3)
        self.lines = {}
        self.colors = {}
        palette = plt.get_cmap("tab10")
        for i, label in enumerate(cases):
            color = palette(i % 10)
            self.colors[label] = color
            self.lines[label] = {
                "fn": self.axes[0].plot([], [], color=color, label=f"{label} F normale")[0],
                "fmeas": self.axes[0].plot([], [], color=color, linestyle="--", alpha=0.45,
                                           label=f"{label} filtrata")[0],
                "ft": self.axes[1].plot([], [], color=color, label=f"{label} tangenziale")[0],
                "fresultant": self.axes[1].plot([], [], color=color, linestyle=":", alpha=0.8,
                                                label=f"{label} risultante")[0],
                "delta": self.axes[2].plot([], [], color=color, label=label)[0],
                "k": self.k_axis.plot([], [], color=color, linestyle="--", alpha=0.5,
                                      label=f"{label} k locale")[0],
                "tau": self.axes[3].plot([], [], color=color, label=label)[0],
            }
        self.axes[0].legend(loc="upper left", fontsize="small", ncol=2)
        self.axes[1].legend(loc="upper left", fontsize="small")
        self.axes[2].legend(loc="upper left", fontsize="small")
        self.k_axis.legend(loc="upper right", fontsize="small")
        self.axes[3].legend(loc="upper left", fontsize="small")
        self.fig.tight_layout()
        if interactive:
            plt.ion()
            plt.show(block=False)

    def update(self, label, sim):
        log = sim.log
        t = np.asarray(log["t"])
        lines = self.lines[label]
        lines["fn"].set_data(t, log["fn"])
        lines["fmeas"].set_data(t, log["fmeas"])
        lines["ft"].set_data(t, log["ft"])
        lines["fresultant"].set_data(t, log["fresultant"])
        lines["delta"].set_data(t, np.asarray(log["delta"]) * 1000)
        lines["k"].set_data(t, log["k"])
        lines["tau"].set_data(t, log["tau"])
        for ax in self.axes:
            ax.relim()
            ax.autoscale_view()
        self.k_axis.relim()
        self.k_axis.autoscale_view()
        self.fig.canvas.draw_idle()
        if self.interactive:
            self.plt.pause(0.001)

    def save(self, path):
        self.fig.savefig(path, dpi=130)
        if self.interactive:
            self.plt.ioff()
            self.plt.show(block=True)
        self.plt.close(self.fig)


def solver_cases():
    cases = []
    for solver_name, solver in (
        ("PGS", mujoco.mjtSolver.mjSOL_PGS),
        ("CG", mujoco.mjtSolver.mjSOL_CG),
        ("Newton", mujoco.mjtSolver.mjSOL_NEWTON),
    ):
        for cone_name, cone in (
            ("pyramidal", mujoco.mjtCone.mjCONE_PYRAMIDAL),
            ("elliptic", mujoco.mjtCone.mjCONE_ELLIPTIC),
        ):
            cases.append((f"{solver_name}-{cone_name}", solver, cone))
    return cases


def run_solver_comparison(duration, outdir, live=True):
    import csv

    cases = solver_cases()
    labels = [case[0] for case in cases]
    if not live:
        import matplotlib
        matplotlib.use("Agg")
    monitor = LiveMonitor(labels, interactive=live)
    outdir.mkdir(exist_ok=True)
    all_rows = []
    summary = []
    for label, solver, cone in cases:
        sim = Sim(solver=solver, cone=cone, contact_model="native")
        print(f"\nProva {label}: solver={solver.name}, cono={cone.name}")
        update_stride = max(1, round(0.02 / sim.dt))
        while sim.data.time < duration:
            for _ in range(update_stride):
                if sim.data.time >= duration:
                    break
                sim.step()
            if live:
                monitor.update(label, sim)
        if not live:
            monitor.update(label, sim)

        log = {key: np.asarray(values) for key, values in sim.log.items()}
        force = log["fn"]
        derivative = np.abs(np.diff(force)) / sim.dt if len(force) > 1 else np.zeros(1)
        contact_mask = log["ncontact"] > 0
        summary.append({
            "solver": label,
            "peak_fn_N": float(force.max(initial=0.0)),
            "max_abs_dFdt_N_s": float(derivative.max(initial=0.0)),
            "mean_fn_contact_N": float(force[contact_mask].mean()) if contact_mask.any() else 0.0,
            "max_penetration_mm": float(log["delta"].max(initial=0.0) * 1000),
            "peak_tangential_N": float(log["ft"].max(initial=0.0)),
            "peak_joint_torque_Nm": float(log["tau"].max(initial=0.0)),
            "contact_samples": int(contact_mask.sum()),
        })
        for i, t in enumerate(log["t"]):
            all_rows.append({
                "solver": label,
                "cone": cone.name,
                **{key: float(values[i]) for key, values in log.items()
                   if key not in ("ncontact",)},
                "ncontact": int(log["ncontact"][i]),
            })
        print("  picco normale={peak_fn_N:.2f} N | max |dF/dt|={max_abs_dFdt_N_s:.0f} N/s | "
              "picco tangenziale={peak_tangential_N:.2f} N | penetrazione max={max_penetration_mm:.3f} mm"
              .format(**summary[-1]))

    csv_path = outdir / "solver_comparison.csv"
    if all_rows:
        with csv_path.open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(all_rows[0]))
            writer.writeheader()
            writer.writerows(all_rows)
    plot_path = outdir / "solver_comparison.png"
    monitor.save(plot_path)
    print(f"\nDati per campione: {csv_path}")
    print(f"Grafici confronto: {plot_path}")
    return summary


def run_live_plot(sim, duration, label="MuJoCo"):
    monitor = LiveMonitor([label])
    Path("out").mkdir(exist_ok=True)
    update_stride = max(1, round(0.02 / sim.dt))
    while sim.data.time < duration:
        for _ in range(update_stride):
            if sim.data.time >= duration:
                break
            sim.step()
        monitor.update(label, sim)
    monitor.save(Path("out") / "live_contact.png")


def run_contact_model_comparison(duration, outdir, descent_time, stiffness_scale, regional_scales):
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(exist_ok=True)
    sim = Sim(
        contact_model="native",
        solver=mujoco.mjtSolver.mjSOL_NEWTON,
        cone=mujoco.mjtCone.mjCONE_ELLIPTIC,
        descent_time=descent_time,
        stiffness_scale=stiffness_scale,
        regional_stiffness_scales=regional_scales,
    )
    while sim.data.time < duration:
        sim.step()

    log = {key: np.asarray(values) for key, values in sim.log.items()}
    t = log["t"]
    native = log["fn"]
    hunt_crossley = log["fhc"]
    contact = log["ncontact"] > 0
    if not contact.any():
        raise RuntimeError("Nessun contatto sonda-tessuto nella durata scelta: impossibile confrontare le forze.")

    contact_indices = np.flatnonzero(contact)
    first_contact = t[contact_indices[0]]
    rmse = float(np.sqrt(np.mean((native[contact] - hunt_crossley[contact]) ** 2)))
    mae = float(np.mean(np.abs(native[contact] - hunt_crossley[contact])))
    peak_native = float(native.max(initial=0.0))
    peak_hc = float(hunt_crossley.max(initial=0.0))
    mean_native = float(native[contact].mean())
    mean_hc = float(hunt_crossley[contact].mean())
    print(f"Primo contatto: {first_contact:.3f} s")
    print(f"Discesa: {sim.descent_time:g} s | velocita' max nominale: "
          f"{1.875 * (HOVER + DEPTH) / sim.descent_time:.4f} m/s")
    print(f"Stiffness globale: {sim.stiffness_scale:g}x | scale zone: {regional_scales}")
    print(f"Fattore solref applicato: {CONTACT_TIME_SCALE:g}x (fisso)")
    print(f"Picco normale MuJoCo: {peak_native:.2f} N | media in contatto: {mean_native:.2f} N")
    print(f"Picco Hunt-Crossley: {peak_hc:.2f} N | media in contatto: {mean_hc:.2f} N")
    print(f"Errore durante contatto: RMSE {rmse:.2f} N | MAE {mae:.2f} N")
    print("Hunt-Crossley e' una stima diagnostica: non viene applicata al Panda.")

    csv_path = outdir / "contact_model_comparison.csv"
    with csv_path.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow([
            "time_s", "contact_count", "mujoco_normal_N", "mujoco_filtered_N",
            "hunt_crossley_N", "hunt_crossley_filtered_N", "penetration_mm",
            "closing_speed_m_s", "local_stiffness_N_m", "solref_timeconst_s",
            "joint_torque_max_Nm", "descent_time_s", "stiffness_scale",
            "k_abdomen_scale", "k_chest_scale", "k_rib_scale", "k_sternum_scale",
            "contact_time_scale",
        ])
        for i in range(len(t)):
            writer.writerow([
                t[i], log["ncontact"][i], native[i], log["fn_lpf"][i],
                hunt_crossley[i], log["fhc_lpf"][i], log["delta"][i] * 1000,
                log["ddelta"][i], log["k"][i], log["solref_tc"][i], log["tau"][i],
                sim.descent_time, sim.stiffness_scale,
                regional_scales["k_abdomen"], regional_scales["k_chest"],
                regional_scales["k_rib"], regional_scales["k_sternum"],
                CONTACT_TIME_SCALE,
            ])

    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.plot(t, native, color="tab:blue", linewidth=1.2, label="MuJoCo: forza normale")
    ax.plot(t, hunt_crossley, color="tab:orange", linewidth=1.2,
            label="Hunt-Crossley: forza stimata")
    ax.axvline(first_contact, color="black", linestyle=":", linewidth=0.9,
               label=f"primo contatto ({first_contact:.2f} s)")
    ax.set_xlim(float(t[0]), float(t[-1]))
    ax.set_xlabel("Tempo simulato [s]")
    ax.set_ylabel("Forza normale [N]")
    zone_label = ", ".join(
        f"{name.removeprefix('k_')}={scale:g}x"
        for name, scale in regional_scales.items() if scale != 1.0
    ) or "zone=1x"
    ax.set_title(
        f"Forza 0–{duration:g} s | discesa {sim.descent_time:g} s | "
        f"k globale {sim.stiffness_scale:g}x ({zone_label}) | solref fisso {CONTACT_TIME_SCALE:g}x "
        f"| RMSE in contatto {rmse:.2f} N"
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    plot_path = outdir / "contact_model_comparison.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Grafico completo 0-{duration:g} s: {plot_path}")
    print(f"Dati: {csv_path}")


def run_headless(sim, duration, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    outdir.mkdir(exist_ok=True)
    m, d = sim.model, sim.data
    pending = [6.0, 10.0, 12.0, 14.0]          # istanti dei fotogrammi da salvare
    frames = []
    t0 = time.time()
    while d.time < duration:
        sim.step()
        if pending and d.time >= pending[0]:
            ps.update_tissue_mesh(m, sim.tissue, d.time)
            sim.color_tip(sim.log["fn"][-1])
            r = mujoco.Renderer(m, 480, 640)
            cam = mujoco.MjvCamera()
            cam.azimuth, cam.elevation, cam.distance = 140, -25, 1.3
            cam.lookat[:] = [0.0, 0.0, 0.15]
            r.update_scene(d, camera=cam)
            frames.append((pending.pop(0), r.render().copy()))
            r.close()
    print(f"simulati {duration:.0f} s in {time.time() - t0:.1f} s di tempo reale")

    L = {k: np.array(v) for k, v in sim.log.items()}
    t, f = L["t"], L["fn"]
    hold = t > sim.descent_time + 2
    dfdt = np.abs(np.diff(f)) / sim.dt
    print(f"F max {f.max():.1f} N | max |dF/dt| {dfdt.max():.0f} N/s | tenuta: F tra {f[hold].min():.1f} e {f[hold].max():.1f} N "
          f"(media {f[hold].mean():.1f}) | coppia max {L['tau'].max():.0f} Nm")

    fig, ax = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    ax[0].plot(t, L["zs"] * 1000, label="superficie sotto la sonda"); ax[0].plot(t, L["ztip"] * 1000, label="punta sonda")
    ax[0].set_ylabel("z [mm]"); ax[0].legend()
    ax[1].plot(t, f, label="forza normale MuJoCo"); ax[1].plot(t, L["fmeas"], lw=0.6, label="sensore filtrato")
    ax[1].axhspan(F_DES - F_TOL, F_DES + F_TOL, color="g", alpha=0.15); ax[1].set_ylabel("F normale [N]"); ax[1].legend()
    ax[2].plot(t, L["delta"] * 1000, label="penetrazione")
    ax[2].set_ylabel("penetrazione [mm]"); ax[2].set_xlabel("t [s]")
    ax[3].plot(t, L["k"], label="k_eq nel contatto")
    ax[3].set_ylabel("rigidita' [N/m]")
    ax3b = ax[3].twinx()
    ax3b.plot(t, L["solref_tc"] * 1000, color="tab:orange", alpha=0.8, label="solref timeconst")
    ax3b.set_ylabel("timeconst [ms]")
    lines = ax[3].get_lines() + ax3b.get_lines()
    ax[3].legend(lines, [line.get_label() for line in lines], loc="upper right")
    for a in ax:
        a.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(outdir / "panda_forza.png", dpi=110)

    frames.sort(key=lambda x: x[0])
    if frames:
        fig2, axs = plt.subplots(2, 2, figsize=(10, 7.5))
        for a, (tt, img) in zip(axs.ravel(), frames):
            a.imshow(img); a.set_title(f"t = {tt:.1f} s"); a.axis("off")
        plt.tight_layout(); plt.savefig(outdir / "panda_frames.png", dpi=100)
    print(f"salvati {outdir}/panda_forza.png e panda_frames.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--random-events", action="store_true")
    ap.add_argument("--vis-gain", type=float, default=1.0, help="esagera SOLO la grafica del respiro (es. 3)")
    ap.add_argument("--live-plot", action="store_true", help="grafici live dei segnali di contatto")
    ap.add_argument("--compare-solvers", action="store_true",
                    help="confronta PGS, CG e Newton con coni compatibili")
    ap.add_argument("--compare-contact-models", action="store_true",
                    help="confronta forza MuJoCo e stima Hunt-Crossley sugli stessi contatti")
    ap.add_argument("--contact-model", choices=("hc", "native"), default="hc",
                    help="hc: forza Hunt-Crossley applicata alla punta; native: contatto MuJoCo")
    ap.add_argument("--solver", choices=("PGS", "CG", "NEWTON"), default="NEWTON")
    ap.add_argument("--cone", choices=("PYRAMIDAL", "ELLIPTIC"), default="ELLIPTIC")
    ap.add_argument("--descent-time", type=float, default=T_DESCENT,
                    help="durata discesa in s; piu' alta = velocita' d'ingresso nominale piu' bassa")
    ap.add_argument("--stiffness-scale", type=float, default=STIFFNESS_SCALE,
                    help="scala di tutti i coefficienti k locali; non modifica CONTACT_TIME_SCALE=40")
    ap.add_argument("--k-abdomen-scale", type=float, default=K_ABDOMEN_SCALE,
                    help="scala aggiuntiva k addome, combinata con --stiffness-scale")
    ap.add_argument("--k-chest-scale", type=float, default=K_CHEST_SCALE,
                    help="scala aggiuntiva k torace, combinata con --stiffness-scale")
    ap.add_argument("--k-rib-scale", type=float, default=K_RIB_SCALE,
                    help="scala aggiuntiva k costole, combinata con --stiffness-scale")
    ap.add_argument("--k-sternum-scale", type=float, default=K_STERNUM_SCALE,
                    help="scala aggiuntiva k sterno, combinata con --stiffness-scale")
    a = ap.parse_args()

    regional_scales = {
        "k_abdomen": a.k_abdomen_scale,
        "k_chest": a.k_chest_scale,
        "k_rib": a.k_rib_scale,
        "k_sternum": a.k_sternum_scale,
    }
    if a.compare_contact_models:
        run_contact_model_comparison(
            a.duration, Path("out"), a.descent_time, a.stiffness_scale, regional_scales,
        )
    elif a.compare_solvers:
        run_solver_comparison(a.duration, Path("out"), live=not a.headless)
    elif a.live_plot:
        solver = {
            "PGS": mujoco.mjtSolver.mjSOL_PGS,
            "CG": mujoco.mjtSolver.mjSOL_CG,
            "NEWTON": mujoco.mjtSolver.mjSOL_NEWTON,
        }[a.solver]
        cone = {
            "PYRAMIDAL": mujoco.mjtCone.mjCONE_PYRAMIDAL,
            "ELLIPTIC": mujoco.mjtCone.mjCONE_ELLIPTIC,
        }[a.cone]
        sim = Sim(random_ev=a.random_events, solver=solver, cone=cone, contact_model=a.contact_model,
                  descent_time=a.descent_time, stiffness_scale=a.stiffness_scale,
                  regional_stiffness_scales=regional_scales)
        if a.headless:
            run_headless(sim, a.duration, Path("out"))
        else:
            run_live_plot(sim, a.duration, f"{a.solver}-{a.cone}")
    else:
        sim = Sim(
            random_ev=a.random_events,
            contact_model=a.contact_model,
            descent_time=a.descent_time,
            stiffness_scale=a.stiffness_scale,
            regional_stiffness_scales=regional_scales,
        )
        if a.headless:
            run_headless(sim, a.duration, Path("out"))
        else:
            run_viewer(sim, a.duration, a.vis_gain)
