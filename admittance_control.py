"""
Controllo di ammettenza a INTEGRATORE PURO per ecografia robotica (Panda in coppia, tessuto simulato).

Implementa la proposta (proposta_strategia_controllo_ammettenza.md):
    M_d * dv/dt + B_d * v = F_virt + (F_des_eff - F_meas)        con K_d -> 0 a contatto
    B_d = 2 * sqrt(M_d * K_t)                                    K_t = rigidita' del TESSUTO nel punto
Con K_d = 0 l'ammettenza e' un integratore puro della forza: a regime F_meas = F_des, errore nullo.

CATENA DEI BLOCCHI (tutta a 1 kHz, un passo fisico = un aggiornamento):
  BLOCCO 0  Misura           F_meas (sensore) e distanza d della punta dalla pelle
  BLOCCO 1  Soglie di K_d    K_d(d, F_meas): alta lontano, scende vicino, si spegne a contatto
  BLOCCO 2  Smorzamento      B_d = 2 sqrt(M_d K_t)     (smorzamento critico del solo tessuto)
  BLOCCO 3  Ammettenza       equazione sopra + integrazione di Eulero -> velocita' v_in (UNA integrazione)
  BLOCCO 4a Traiettoria      linea in XY lungo il busto (parte a contatto stabilizzato, si ferma se si perde il contatto)
  BLOCCO 4  Velocita'        v_in lungo la normale + velocita' tangente (feedforward + P) + P di orientamento -> v6 (6D)
  BLOCCO 5  IK differenziale q_dot = J^+ v6 + nullspace;   q_ref += q_dot * dt
  BLOCCO 6  Posizione        PD di giunto in coppia verso q_ref + compensazione di gravita'

Convenzione di segno: v_in > 0 = la punta si muove VERSO la pelle (asse -n, con n normale uscente).
L'errore di forza e' e_F = F_des_eff - F_meas: positivo = forza troppo bassa = il robot deve avanzare.
"""
from dataclasses import dataclass
import numpy as np
import mujoco

import panda_scene as ps
from tissue import Tissue, min_jerk, min_jerk_dot
from ft_sensor import ATINanoSensor


@dataclass
class AdmittanceParams:
    F_DES: float = 5.0             # forza di contatto desiderata [N]

    K_FAR: float = 1500.0          # rigidezza virtuale lontano dalla pelle [N/m]
    K_NEAR: float = 300.0          # rigidezza virtuale alla soglia D_NEAR [N/m]
    D_FAR: float = 0.060           # [m] d >= D_FAR          -> K_d = K_FAR
    D_NEAR: float = 0.020          # [m] D_NEAR <= d < D_FAR -> K_d scende da K_FAR a K_NEAR
    D_OFF: float = 0.000           # [m] D_OFF <= d < D_NEAR  -> K_d scende da K_NEAR a 0; d <= D_OFF -> K_d = 0
    F_GATE0: float = 0.5           # [N] soglia di forza sopra cui K_d inizia a spegnersi
    F_GATE1: float = 2.0           # [N] soglia di forza sopra cui K_d e' spento del tutto

    M_D: float = 0.05              # massa virtuale [kg]
    KT_MODE: str = "k_eq"          # "k_eq": rigidezza della mappa del tessuto | "tangent": pendenza al punto di lavoro
    B_INCLUDE_KD: bool = False     # False: B_d = 2 sqrt(M K_t) (proposta) | True: 2 sqrt(M (K_t + K_d)) anche in avvicinamento
    B_MIN: float = 1.0             # [Ns/m] limite inferiore di sicurezza

    INTEGRATOR: str = "implicit"   # "explicit" | "implicit" (Eulero esplicito o implicito sul termine di smorzamento)

    V_CRUISE: float = 0.10         # velocita' massima di avvicinamento lontano dalla pelle [m/s]
    V_TOUCH: float = 0.02          # velocita' massima di avvicinamento entro D_NEAR [m/s]
    V_CONTACT: float = 0.02        # velocita' massima verso la pelle quando c'e' contatto [m/s]
    V_RETRACT: float = 0.10        # velocita' massima di allontanamento [m/s]

    SCAN_ENABLE: bool = True
    SCAN_PATTERN: str = "serpentine"    # "line" mantiene il percorso singolo definito da SCAN_START/END
    SCAN_X_RANGE: tuple = (-0.16, 0.16) # [m] limiti cranio-caudali, con margine dal bordo
    SCAN_Y_RANGE: tuple = (-0.11, 0.11) # [m] limiti trasversali, con margine dal bordo
    SCAN_ROW_SPACING: float = 0.04      # [m] distanza massima tra le passate
    SCAN_START: tuple = (0.04, 0.06)    # [m] inizio della linea singola
    SCAN_END: tuple = (-0.08, 0.06)     # [m] fine della linea singola
    SCAN_SPEED: float = 0.015           # [m/s] velocita' MEDIA sul percorso, con profilo min-jerk per segmento
    SCAN_DELAY: float = 2.0             # [s] attesa dopo il primo contatto prima di partire
    F_CONTACT_OK: float = 1.0           # [N] il clock della traiettoria avanza solo se F_meas supera questa soglia

    KP_POS_V: float = 4.0          # [1/s] guadagno P posizione tangenziale -> velocita'
    KP_ROT_V: float = 4.0          # [1/s] guadagno P orientamento -> velocita' angolare
    V_TAN_MAX: float = 0.10        # [m/s]
    W_MAX: float = 0.5             # [rad/s]

    IK_DAMPING: float = 0.02       # smorzamento lambda della pseudo-inversa (adimensionale)
    K_NULL_V: float = 1.0          # [1/s] richiamo alla postura nel nullspace
    QDOT_MAX: float = 1.0          # [rad/s] limite sulla velocita' di ogni giunto
    Q_REF_MAX_ERR: float = 0.08    # [rad] anti-windup: |q_ref - q| limitato

    KP_JOINT: tuple = (600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0)   # [Nm/rad] (valori Franka)
    ZETA_JOINT: float = 1.0        # smorzamento relativo del PD di giunto (adimensionale)


class StiffnessSchedule:
    """
    K_d(d) con transizioni C2 (min-jerk), cioe' senza scalini ne' nella K ne' nelle sue derivate:
        d >= D_FAR            K_d = K_FAR                                   (robot rigido, preciso)
        D_NEAR <= d < D_FAR   K_d: K_FAR  -> K_NEAR   (fase "a)")
        D_OFF  <= d < D_NEAR  K_d: K_NEAR -> 0        (fase "b)": spegnimento graduale)
        d <= D_OFF            K_d = 0                 (integratore puro)

    Formula:  K_d = [K_NEAR + (K_FAR - K_NEAR) * s_a(d)] * s_b(d) * gate(F_meas)
    s_a e s_b sono min-jerk: 0 -> 1 con derivata prima e seconda nulle agli estremi.
    """

    def __init__(self, p: AdmittanceParams):
        self.p = p

    def s_a(self, d):
        return float(min_jerk((d - self.p.D_NEAR) / (self.p.D_FAR - self.p.D_NEAR)))

    def s_b(self, d):
        return float(min_jerk((d - self.p.D_OFF) / (self.p.D_NEAR - self.p.D_OFF)))

    def force_gate(self, F_meas):
        """1 senza contatto, 0 con contatto stabilito (forza misurata > F_GATE1)."""
        p = self.p
        return 1.0 - float(min_jerk((F_meas - p.F_GATE0) / (p.F_GATE1 - p.F_GATE0)))

    def K(self, d, F_meas=0.0):
        p = self.p
        k_dist = (p.K_NEAR + (p.K_FAR - p.K_NEAR) * self.s_a(d)) * self.s_b(d)
        return k_dist * self.force_gate(F_meas)

    def rho(self, d, F_meas=0.0):
        """Quanto della forza desiderata e' attivo (0 lontano, 1 a contatto): subentra mentre K_d si spegne."""
        return max(1.0 - self.s_b(d), 1.0 - self.force_gate(F_meas))

    def v_in_max(self, d, F_meas=0.0):
        """Velocita' massima verso la pelle: scende a V_TOUCH vicino alla pelle; con contatto vale V_CONTACT."""
        p = self.p
        if F_meas > p.F_GATE1:
            return p.V_CONTACT
        return p.V_TOUCH + (p.V_CRUISE - p.V_TOUCH) * self.s_a(d)


def tissue_stiffness(tissue, x, y, F_des, mode="k_eq"):
    """
    K_t [N/m] del tessuto nel punto (x, y).
      "k_eq"    : valore della mappa (pendenza dF/d(delta) a delta_ref = 8 mm)
      "tangent" : pendenza dF/d(delta) del modello Hunt-Crossley F = k d^n al punto di lavoro F = F_des.
                  d* = (F_des / k)^(1/n)  ->  K_t = n * F_des / d*      (= k_eq * sqrt(d*/delta_ref): circa 0.9 k_eq a 5 N)
    """
    k_eq = float(tissue.stiff(x, y))
    if mode == "k_eq":
        return k_eq
    n = tissue.n
    k = k_eq / (n * tissue.delta_ref ** (n - 1.0))
    d_star = (F_des / k) ** (1.0 / n)
    return n * F_des / d_star


def critical_damping(M_d, K, B_min=0.0):
    """B_d = 2 sqrt(M_d K): smorzamento critico di un sistema massa-molla M ddx + B dx + K x = 0."""
    return max(2.0 * np.sqrt(M_d * K), B_min)


class PureIntegratorAdmittance:
    """
        M_d dv/dt + B_d v = F_virt + e_F        (v = velocita' verso la pelle)

    Con K_d = 0 (F_virt = 0) e' un integratore della forza: H(s) = 1 / (s (M_d s + B_d)) per la posizione.
    Si integra UNA volta per avere la velocita' (la posizione si ottiene dopo, in giunto, con q_ref += q_dot dt).

    Eulero esplicito :  v[k+1] = v[k] + dt/M * (F_net - B v[k])
    Eulero implicito :  v[k+1] = (M v[k] + dt F_net) / (M + dt B)     (incondizionatamente stabile in B)
    Con dt = 1 ms e costante di tempo M_d/B_d di 3-5 ms entrambi gli schemi sono stabili.
    """

    def __init__(self, M_d, dt, integrator="implicit"):
        self.M, self.dt, self.integrator = M_d, dt, integrator
        self.v = 0.0

    def step(self, F_net, B_d):
        if self.integrator == "explicit":
            self.v = self.v + self.dt / self.M * (F_net - B_d * self.v)
        else:
            self.v = (self.M * self.v + self.dt * F_net) / (self.M + self.dt * B_d)
        return self.v

    def clamp(self, v_min, v_max):
        """Limita la velocita' e RIPORTA nello stato il valore limitato (anti-windup per back-calculation)."""
        self.v = float(np.clip(self.v, v_min, v_max))
        return self.v


class LineTrajectory:
    """
    Percorso XY con profilo min-jerk su ogni segmento.
    Il clock avanza solo con contatto stabilizzato e si ferma se la forza cala.
    """

    def __init__(self, p: AdmittanceParams):
        self.p = p
        if p.SCAN_SPEED <= 0 or p.SCAN_ROW_SPACING <= 0:
            raise ValueError("SCAN_SPEED e SCAN_ROW_SPACING devono essere positivi")
        if p.SCAN_PATTERN == "line":
            self.waypoints = np.array([p.SCAN_START, p.SCAN_END], dtype=float)
        elif p.SCAN_PATTERN == "serpentine":
            x_min, x_max = p.SCAN_X_RANGE
            y_min, y_max = p.SCAN_Y_RANGE
            if x_min >= x_max or y_min >= y_max:
                raise ValueError("I limiti SCAN_X_RANGE e SCAN_Y_RANGE devono essere crescenti")
            rows = int(np.ceil((y_max - y_min) / p.SCAN_ROW_SPACING)) + 1
            y_values = np.linspace(y_min, y_max, rows)
            self.waypoints = np.array([
                (x, y)
                for row, y in enumerate(y_values)
                for x in ((x_min, x_max) if row % 2 == 0 else (x_max, x_min))
            ], dtype=float)
        else:
            raise ValueError(f"SCAN_PATTERN non supportato: {p.SCAN_PATTERN}")
        segment_lengths = np.linalg.norm(np.diff(self.waypoints, axis=0), axis=1)
        if np.any(segment_lengths <= 0):
            raise ValueError("Il percorso di scansione contiene segmenti di lunghezza nulla")
        self.segment_durations = segment_lengths / p.SCAN_SPEED
        self.segment_end_times = np.cumsum(self.segment_durations)
        self.length = float(segment_lengths.sum())
        self.T = float(self.segment_end_times[-1])
        self.clock = 0.0
        self.t_contact = None

    def update(self, t, dt, F_meas):
        p = self.p
        if self.t_contact is None and F_meas > p.F_GATE1:
            self.t_contact = t
        if self.t_contact is not None and t - self.t_contact >= p.SCAN_DELAY and F_meas > p.F_CONTACT_OK:
            self.clock = min(self.clock + dt, self.T)
        if self.clock >= self.T:
            return self.waypoints[-1].copy(), np.zeros(2), 1.0
        segment = int(np.searchsorted(self.segment_end_times, self.clock, side="right"))
        segment_start = self.segment_end_times[segment - 1] if segment else 0.0
        duration = self.segment_durations[segment]
        s = (self.clock - segment_start) / duration
        delta = self.waypoints[segment + 1] - self.waypoints[segment]
        xy = self.waypoints[segment] + delta * float(min_jerk(s))
        ds_dt = float(min_jerk_dot(s)) / duration
        return xy, delta * ds_dt, self.clock / self.T


def differential_ik(J, v6, q, q_null, p: AdmittanceParams):
    """q_dot = J^+ v6 + (I - J^+ J) k_null (q_null - q), con pseudo-inversa smorzata (DLS)."""
    nv = J.shape[1]
    J_pinv = J.T @ np.linalg.inv(J @ J.T + p.IK_DAMPING ** 2 * np.eye(6))
    q_dot = J_pinv @ v6 + (np.eye(nv) - J_pinv @ J) @ (p.K_NULL_V * (q_null - q))
    return np.clip(q_dot, -p.QDOT_MAX, p.QDOT_MAX)


HOVER = 0.06          # quota di partenza sopra la pelle a riposo [m]
GOAL_DEPTH = 0.008    # profondita' dell'obiettivo del planner sotto la pelle a riposo [m]


class AdmittanceSim:
    """
    Panda in coppia + sonda + tessuto Hunt-Crossley + sensore F/T, con la legge di controllo:
    ammettenza ad integratore puro -> IK differenziale -> PD di posizione di giunto.
    Ogni chiamata a step() e' un passo fisico da 1 ms; i segnali finiscono in self.log.
    """

    def __init__(self, params: AdmittanceParams = None, ati_calibration="SI-25-0.25",
                 ati_filter=None, descent_time=4.0, goal_depth=GOAL_DEPTH, seed=0):
        if descent_time <= 0:
            raise ValueError("descent_time deve essere positivo")
        self.p = params or AdmittanceParams()
        self.traj = LineTrajectory(self.p)
        self.x0, self.y0 = float(self.traj.waypoints[0][0]), float(self.traj.waypoints[0][1])
        self.descent_time, self.goal_depth = float(descent_time), float(goal_depth)

        self.model, self.ids = ps.build_model()
        self.data = mujoco.MjData(self.model)
        m, d = self.model, self.data
        self.dt = m.opt.timestep
        self.tissue = Tissue()

        mujoco.mj_resetDataKeyframe(m, d, m.key("hover").id)
        self.R_des = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1.0]])
        self.z_rest = float(self.tissue.geom.z0(self.x0, self.y0))
        self.q_null = d.qpos[:m.nv].copy()
        mujoco.mj_forward(m, d)
        start_err = np.linalg.norm(d.site_xpos[self.ids["site"]] - np.array([self.x0, self.y0, self.z_rest + HOVER]))
        if start_err > 1e-3:
            raise ValueError(f"la posa salvata ps.Q_HOVER non e' {HOVER*100:.0f} cm sopra il punto di partenza "
                             f"({self.x0:.3f}, {self.y0:.3f}): scarto {start_err*1000:.1f} mm. Va ricalcolata.")

        self.schedule = StiffnessSchedule(self.p)
        self.adm = PureIntegratorAdmittance(self.p.M_D, self.dt, self.p.INTEGRATOR)
        self.q_ref = d.qpos[:m.nv].copy()
        self.kp_joint = np.array(self.p.KP_JOINT)

        self.ft = ATINanoSensor(self.dt, ati_calibration, ati_filter, seed=seed)
        d.ctrl[:] = d.qfrc_bias[:m.nv]
        mujoco.mj_forward(m, d)
        self._ft_raw = d.sensordata[self.ids["ft_adr"]:self.ids["ft_adr"] + 6].copy()
        self.ft.tare(self._ft_raw)

        self.log = {k: [] for k in (
            "t", "fn", "fmeas", "delta", "ncontact", "zs", "ztip", "x", "y", "tau", "ft_saturated",
            "Kd", "Bd", "Kt", "v_in", "v_lim", "dist", "Fdes_eff", "F_virt",
            "x_ref", "y_ref", "z_ref", "scan_s", "exy")}

    def target(self, t):
        """Obiettivo del planner: discesa min-jerk da HOVER sopra la pelle a goal_depth sotto, in descent_time."""
        z0 = self.z_rest + HOVER
        z1 = self.z_rest - self.goal_depth
        return np.array([self.x0, self.y0, z0 + (z1 - z0) * min_jerk(t / self.descent_time)])

    def step(self):
        m, d, ids, p = self.model, self.data, self.ids, self.p
        t, dt = d.time, self.dt
        mujoco.mj_step1(m, d)

        pos = d.site_xpos[ids["site"]].copy()
        R = d.site_xmat[ids["site"]].reshape(3, 3)
        J = ps.jac(m, d, ids["site"])
        qd = d.qvel[:m.nv]
        v_tip = J @ qd
        hc = self.tissue.contact(pos, v_tip[:3], t)
        ps.apply_tip_force(d, ids["probe_body"], hc.force, pos)
        n = hc.normal
        dist = -hc.delta

        w_meas = self.ft.read(self._ft_raw)
        R_ft = d.site_xmat[ids["ft_site"]].reshape(3, 3)
        F_meas = float(-(R_ft @ w_meas[:3]) @ n)

        K_d = self.schedule.K(dist, F_meas)
        rho = self.schedule.rho(dist, F_meas)
        F_des_eff = rho * p.F_DES

        # Smorzamento critico B_d = 2 sqrt(M_d K_t). Qui K_t e' la rigidezza VERA del tessuto nel punto (x, y),
        # nota solo in simulazione. Nella fase successiva (SAC) K_t verra' sostituita da una stima unica e
        # costante K_hat, come sul robot reale: B_nom = 2 sqrt(M_d K_hat) si calcola una volta sola e l'agente
        # SAC lo modula di +-20 % per compensare l'errore della stima; lo stesso agente somma una forza
        # correttiva all'ingresso dell'ammettenza (con saturazione sulla somma delle forze).
        K_t = tissue_stiffness(self.tissue, pos[0], pos[1], p.F_DES, p.KT_MODE)
        B_d = critical_damping(p.M_D, K_t + (K_d if p.B_INCLUDE_KD else 0.0), p.B_MIN)

        goal = self.target(t)
        if p.SCAN_ENABLE:
            xy_ref, v_ff_xy, scan_s = self.traj.update(t, dt, F_meas)
            goal[0], goal[1] = xy_ref
        else:
            v_ff_xy, scan_s = np.zeros(2), 0.0

        e_in = float((goal - pos) @ (-n))
        F_virt = K_d * e_in
        e_F = F_des_eff - F_meas
        v_in = self.adm.step(F_virt + e_F, B_d)
        v_lim = self.schedule.v_in_max(dist, F_meas)
        v_in = self.adm.clamp(-p.V_RETRACT, v_lim)

        e_xy = np.array([goal[0] - pos[0], goal[1] - pos[1]])
        v_xy = v_ff_xy + p.KP_POS_V * e_xy
        v_tan = np.array([v_xy[0], v_xy[1], -(n[0] * v_xy[0] + n[1] * v_xy[1]) / n[2]])
        v_tan = v_tan * min(1.0, p.V_TAN_MAX / max(np.linalg.norm(v_tan), 1e-9))
        w = np.clip(p.KP_ROT_V * ps.rot_err(R, self.R_des), -p.W_MAX, p.W_MAX)
        v6 = np.concatenate([-n * v_in + v_tan, w])

        q = d.qpos[:m.nv]
        q_dot_cmd = differential_ik(J, v6, q, self.q_null, p)
        self.q_ref = self.q_ref + q_dot_cmd * dt
        self.q_ref = q + np.clip(self.q_ref - q, -p.Q_REF_MAX_ERR, p.Q_REF_MAX_ERR)

        M = np.zeros((m.nv, m.nv))
        mujoco.mj_fullM(m, d, M)
        kd_joint = 2.0 * p.ZETA_JOINT * np.sqrt(self.kp_joint * np.diag(M))
        tau = self.kp_joint * (self.q_ref - q) + kd_joint * (q_dot_cmd - qd)
        d.ctrl[:] = np.clip(tau + d.qfrc_bias[:m.nv], -ps.TORQUE_LIMITS, ps.TORQUE_LIMITS)
        mujoco.mj_step2(m, d)
        self._ft_raw = d.sensordata[ids["ft_adr"]:ids["ft_adr"] + 6].copy()

        for k, val in (("t", t), ("fn", hc.fn), ("fmeas", F_meas), ("delta", max(hc.delta, 0.0)),
                       ("ncontact", hc.fn > 1e-3), ("zs", hc.zs), ("ztip", pos[2]), ("x", pos[0]), ("y", pos[1]),
                       ("tau", np.abs(d.ctrl).max()), ("ft_saturated", self.ft.saturated),
                       ("Kd", K_d), ("Bd", B_d), ("Kt", K_t), ("v_in", v_in), ("v_lim", v_lim), ("dist", dist),
                       ("Fdes_eff", F_des_eff), ("F_virt", F_virt), ("x_ref", goal[0]), ("y_ref", goal[1]),
                       ("z_ref", goal[2]), ("scan_s", scan_s), ("exy", np.hypot(*e_xy))):
            self.log[k].append(float(val))
        return hc.fn
