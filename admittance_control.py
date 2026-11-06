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
from dataclasses import dataclass, field
import numpy as np
import mujoco

import demo_panda as dp
import panda_scene as ps
from tissue import min_jerk


# =====================================================================================================
#  PARAMETRI  (tutto quello che puoi manipolare e' qui)
# =====================================================================================================
@dataclass
class AdmittanceParams:
    # --- forza obiettivo --------------------------------------------------------------------------
    F_DES: float = 5.0             # forza di contatto desiderata [N]

    # --- BLOCCO 1: soglie di K_d (distanza d dalla pelle, in metri, positiva = sopra la pelle) ------
    K_FAR: float = 1500.0          # rigidezza virtuale lontano dalla pelle [N/m]
    K_NEAR: float = 300.0          # rigidezza virtuale alla soglia D_NEAR [N/m]
    D_FAR: float = 0.060           # d >= D_FAR          -> K_d = K_FAR
    D_NEAR: float = 0.020          # D_NEAR <= d < D_FAR -> K_d scende da K_FAR a K_NEAR
    D_OFF: float = 0.000           # D_OFF <= d < D_NEAR -> K_d scende da K_NEAR a 0; d <= D_OFF -> K_d = 0
    F_GATE0: float = 0.5           # rete di sicurezza in forza: sopra F_GATE0 K_d inizia a spegnersi...
    F_GATE1: float = 2.0           # ...e sopra F_GATE1 e' completamente spento (anche se la stima di d e' sbagliata)

    # --- BLOCCO 2: smorzamento --------------------------------------------------------------------
    M_D: float = 0.05              # massa virtuale [kg]  (10 g: ammettenza molto reattiva; vedi nota su rumore e banda)
    KT_MODE: str = "k_eq"          # "k_eq": K_t = k_eq(x,y) della tua mappa (come nella proposta)
                                   # "tangent": pendenza dF/d(delta) del modello Hunt-Crossley al punto di lavoro F_DES
                                   #            (circa 1.5 k_eq: smorzamento piu' alto, vedi tissue_stiffness)
    B_INCLUDE_KD: bool = False     # False: B_d = 2 sqrt(M K_t) (proposta) | True: 2 sqrt(M (K_t + K_d)) anche in avvicinamento
    B_MIN: float = 1.0             # [Ns/m] limite inferiore di sicurezza

    # --- BLOCCO 3: integratore ---------------------------------------------------------------------
    INTEGRATOR: str = "implicit"   # "explicit" | "implicit" (Eulero esplicito o implicito sul termine di smorzamento)

    # --- limiti di velocita' (lungo la normale) ----------------------------------------------------
    V_CRUISE: float = 0.10         # velocita' massima di avvicinamento lontano dalla pelle [m/s]
    V_TOUCH: float = 0.02          # velocita' massima di avvicinamento entro D_NEAR [m/s]
    V_CONTACT: float = 0.06        # velocita' massima verso la pelle quando c'e' contatto [m/s]
    V_RETRACT: float = 0.10        # velocita' massima di allontanamento [m/s]

    # --- BLOCCO 4a: traiettoria in XY --------------------------------------------------------------
    SCAN_ENABLE: bool = True
    SCAN_PATTERN: str = "serpentine"    # "line" mantiene il percorso singolo definito da SCAN_START/END
    SCAN_X_RANGE: tuple = (-0.16, 0.16) # [m] limiti cranio-caudali, con margine dal bordo
    SCAN_Y_RANGE: tuple = (-0.11, 0.11) # [m] limiti trasversali, con margine dal bordo
    SCAN_ROW_SPACING: float = 0.04      # [m] distanza massima tra le passate
    SCAN_START: tuple = (0.04, 0.06)    # [m] inizio/fine della linea singola
    SCAN_END: tuple = (-0.08, 0.06)    # [m] fine/inizio della linea singola
    SCAN_SPEED: float = 0.015           # [m/s] velocita' MEDIA sul percorso, con profilo min-jerk per segmento
    SCAN_DELAY: float = 2.0             # [s] attesa dopo il primo contatto prima di partire
    F_CONTACT_OK: float = 1.0           # [N] il clock della traiettoria avanza solo se F_meas supera questa soglia

    # --- BLOCCO 4: assi non controllati in forza ---------------------------------------------------
    KP_POS_V: float = 4.0          # [1/s] guadagno P posizione tangenziale -> velocita'
    KP_ROT_V: float = 4.0          # [1/s] guadagno P orientamento -> velocita' angolare
    V_TAN_MAX: float = 0.10        # [m/s]
    W_MAX: float = 0.5             # [rad/s]

    # --- BLOCCO 5: IK differenziale ----------------------------------------------------------------
    IK_DAMPING: float = 0.02       # lambda della pseudo-inversa smorzata
    K_NULL_V: float = 1.0          # [1/s] richiamo alla postura nel nullspace
    QDOT_MAX: float = 1.0          # [rad/s] limite sulla velocita' di ogni giunto
    Q_REF_MAX_ERR: float = 0.08    # [rad] anti-windup: |q_ref - q| limitato

    # --- BLOCCO 6: controllore di posizione di giunto ---------------------------------------------
    KP_JOINT: tuple = (600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0)   # [Nm/rad] (valori Franka)
    ZETA_JOINT: float = 1.0        # Kd = 2 zeta sqrt(Kp M_ii): smorzamento critico per giunto


# =====================================================================================================
#  BLOCCO 1: SOGLIE DI K_d  (qui si vede come K scala con la distanza e si spegne)
# =====================================================================================================
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


# =====================================================================================================
#  BLOCCO 2: SMORZAMENTO CRITICO DEL TESSUTO
# =====================================================================================================
def tissue_stiffness(tissue, x, y, F_des, mode="tangent"):
    """
    K_t [N/m] del tessuto nel punto (x, y).
      "k_eq"    : valore della mappa (rigidezza equivalente lineare a delta_ref = 8 mm)
      "tangent" : pendenza dF/d(delta) del modello Hunt-Crossley F = k d^n al punto di lavoro F = F_des.
                  d* = (F_des / k)^(1/n)  ->  K_t = n * F_des / d*      (con n = 1.5 e' circa 1.5 k_eq)
    """
    k_eq = float(tissue.stiff(x, y))
    if mode == "k_eq":
        return k_eq
    n = tissue.n
    k = k_eq * tissue.delta_ref ** (1.0 - n)
    d_star = (F_des / k) ** (1.0 / n)
    return n * F_des / d_star


def critical_damping(M_d, K, B_min=0.0):
    """B_d = 2 sqrt(M_d K): smorzamento critico di un sistema massa-molla M ddx + B dx + K x = 0."""
    return max(2.0 * np.sqrt(M_d * K), B_min)


# =====================================================================================================
#  BLOCCO 3: AMMETTENZA (integratore) + EULERO
# =====================================================================================================
class PureIntegratorAdmittance:
    """
        M_d dv/dt + B_d v = F_virt + e_F        (v = velocita' verso la pelle)

    Con K_d = 0 (F_virt = 0) e' un integratore della forza: H(s) = 1 / (s (M_d s + B_d)) per la posizione.
    Si integra UNA volta per avere la velocita' (la posizione si ottiene dopo, in giunto, con q_ref += q_dot dt).

    Eulero esplicito :  v[k+1] = v[k] + dt/M * (F_net - B v[k])
    Eulero implicito :  v[k+1] = (M v[k] + dt F_net) / (M + dt B)     (incondizionatamente stabile in B)
    Con dt = 1 ms e costante di tempo M/B ~ 20 ms i due schemi differiscono pochissimo.
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


# =====================================================================================================
#  BLOCCO 4a: TRAIETTORIA IN XY
# =====================================================================================================
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
        ds_dt = (30 * s ** 2 - 60 * s ** 3 + 30 * s ** 4) / duration
        return xy, delta * ds_dt, self.clock / self.T


# =====================================================================================================
#  BLOCCO 5: CINEMATICA INVERSA DIFFERENZIALE
# =====================================================================================================
def differential_ik(J, v6, q, q_null, p: AdmittanceParams):
    """q_dot = J^+ v6 + (I - J^+ J) k_null (q_null - q), con pseudo-inversa smorzata (DLS)."""
    nv = J.shape[1]
    J_pinv = J.T @ np.linalg.inv(J @ J.T + p.IK_DAMPING ** 2 * np.eye(6))
    q_dot = J_pinv @ v6 + (np.eye(nv) - J_pinv @ J) @ (p.K_NULL_V * (q_null - q))
    return np.clip(q_dot, -p.QDOT_MAX, p.QDOT_MAX)


# =====================================================================================================
#  SIMULAZIONE CON IL CONTROLLO DI AMMETTENZA
# =====================================================================================================
class AdmittanceSim(dp.Sim):
    """
    Come dp.Sim (tessuto, sensore, log, viewer) ma con la legge di controllo sostituita:
    ammettenza ad integratore puro -> IK differenziale -> PD di posizione di giunto.
    """

    def __init__(self, params: AdmittanceParams = None, contact_model="hc", **kw):
        if contact_model not in ("hc", "native"):
            raise ValueError("contact_model deve essere 'hc' o 'native'")
        params = params or AdmittanceParams()
        trajectory = LineTrajectory(params)
        if params.SCAN_ENABLE:
            dp.X0, dp.Y0 = trajectory.waypoints[0]               # la discesa avviene sul primo punto del percorso
        super().__init__(contact_model=contact_model, **kw)     # il genitore prepara collisioni e corpo della sonda
        self.p = params
        self.traj = trajectory
        self.schedule = StiffnessSchedule(self.p)
        self.adm = PureIntegratorAdmittance(self.p.M_D, self.dt, self.p.INTEGRATOR)
        m, d = self.model, self.data
        self.q_ref = d.qpos[:m.nv].copy()                       # riferimento di giunto (integrato da q_dot)
        self.kp_joint = np.array(self.p.KP_JOINT)
        self._fn_prev = 0.0                                     # forza dell'istante precedente (solo contatto nativo)
        for k in ("Kd", "Bd", "Kt", "v_in", "dist", "Fdes_eff", "F_virt", "v_lim",
                  "y", "x_ref", "y_ref", "z_ref", "scan_s", "exy"):
            self.log[k] = []
        dp.F_DES = self.p.F_DES                                 # colore della punta nel viewer

    def step(self):
        m, d, ids, p = self.model, self.data, self.ids, self.p
        native = self.contact_model == "native"
        t, dt = d.time, self.dt
        if native:
            ps.update_tissue_mesh(m, self.tissue, t)
            k_candidate, tc_candidate = self.configure_contact_stiffness()
        mujoco.mj_step1(m, d)

        # ---------------------------------------------------------------- BLOCCO 0: misura
        pos = d.site_xpos[ids["site"]].copy()
        R = d.site_xmat[ids["site"]].reshape(3, 3)
        compression = ps.measure_tissue_compression(m, d, ids, self.tissue, t)
        J = dp.jac(m, d, ids["site"])
        qd = d.qvel[:m.nv]
        v_tip = J @ qd
        if not native:                                          # forza del tessuto (Hunt-Crossley) sulla punta
            hc = self.tissue.contact(pos, v_tip[:3], t)
            d.qfrc_applied[:] = 0.0
            mujoco.mj_applyFT(m, d, hc.force, np.zeros(3), pos, self.body, d.qfrc_applied)
            fn_true = hc.fn
        else:
            fn_true = self._fn_prev
        F_meas = self.sensor.read(fn_true)                      # UNICO segnale di forza visto dal controllore

        n = self.tissue.normal(pos[0], pos[1], t)               # normale uscente della pelle sotto la punta
        zs = self.tissue.surface_height(pos[0], pos[1], t)
        dist = float((pos[2] - zs) * n[2])                      # distanza dalla pelle lungo n (>0 = sopra)

        # ---------------------------------------------------------------- BLOCCO 1: soglie di K_d
        K_d = self.schedule.K(dist, F_meas)
        rho = self.schedule.rho(dist, F_meas)
        F_des_eff = rho * p.F_DES

        # ---------------------------------------------------------------- BLOCCO 2: smorzamento critico
        K_t = tissue_stiffness(self.tissue, pos[0], pos[1], p.F_DES, p.KT_MODE)
        B_d = critical_damping(p.M_D, K_t + (K_d if p.B_INCLUDE_KD else 0.0), p.B_MIN)

        # ---------------------------------------------------------------- BLOCCO 3: ammettenza + Eulero
        goal = self.target(t)                                   # obiettivo del planner (serve solo finche' K_d > 0)
        # ---------------------------------------------------------------- BLOCCO 4a: traiettoria XY
        if p.SCAN_ENABLE:
            xy_ref, v_ff_xy, scan_s = self.traj.update(t, dt, F_meas)
            goal[0], goal[1] = xy_ref                           # la quota z del planner resta quella della discesa
        else:
            v_ff_xy, scan_s = np.zeros(2), 0.0
        e_in = float((goal - pos) @ (-n))                       # distanza dall'obiettivo lungo l'asse verso la pelle
        F_virt = K_d * e_in                                     # forza virtuale della molla (0 a contatto)
        e_F = F_des_eff - F_meas                                # errore di forza
        self.adm.M = p.M_D
        v_in = self.adm.step(F_virt + e_F, B_d)
        v_lim = self.schedule.v_in_max(dist, F_meas)
        v_in = self.adm.clamp(-p.V_RETRACT, v_lim)

        # ---------------------------------------------------------------- BLOCCO 4: velocita' cartesiana 6D
        # velocita' TANGENTE alla pelle: feedforward della linea + correzione P sull'errore in XY.
        # La componente z si ricava imponendo v_tan . n = 0, cosi' la sonda segue la curvatura del busto.
        e_xy = np.array([goal[0] - pos[0], goal[1] - pos[1], 0.0])
        v_xy = v_ff_xy + p.KP_POS_V * e_xy[:2]
        v_tan = np.array([v_xy[0], v_xy[1], -(n[0] * v_xy[0] + n[1] * v_xy[1]) / n[2]])
        v_tan = v_tan * min(1.0, p.V_TAN_MAX / max(np.linalg.norm(v_tan), 1e-9))
        w = np.clip(p.KP_ROT_V * dp.rot_err(R, self.R_des), -p.W_MAX, p.W_MAX)
        v6 = np.concatenate([-n * v_in + v_tan, w])

        # ---------------------------------------------------------------- BLOCCO 5: IK differenziale
        q = d.qpos[:m.nv]
        q_dot_cmd = differential_ik(J, v6, q, self.q_null, p)
        self.q_ref = self.q_ref + q_dot_cmd * dt                # integrazione in giunto -> riferimento di posizione
        self.q_ref = q + np.clip(self.q_ref - q, -p.Q_REF_MAX_ERR, p.Q_REF_MAX_ERR)   # anti-windup

        # ---------------------------------------------------------------- BLOCCO 6: PD di posizione di giunto
        Minv = np.zeros((m.nv, m.nv))
        mujoco.mj_solveM(m, d, Minv, np.eye(m.nv))
        M_diag = np.diag(np.linalg.inv(Minv))
        kd_joint = 2.0 * p.ZETA_JOINT * np.sqrt(self.kp_joint * M_diag)
        tau = self.kp_joint * (self.q_ref - q) + kd_joint * (q_dot_cmd - qd)
        d.ctrl[:] = np.clip(tau + d.qfrc_bias[:m.nv], -ps.TORQUE_LIMITS, ps.TORQUE_LIMITS)
        self._J_pos = J[:3].copy()
        self.last_F_cmd = np.zeros(6)
        mujoco.mj_step2(m, d)

        # ---------------------------------------------------------------- log (stesse chiavi di dp.Sim + le nuove)
        if native:
            fn, ft, fres, pen, k_eq, tc, ncon = self.contact_force()
            fhc, closing = self.hunt_crossley_force()
            tau_ext = d.qfrc_constraint[:m.nv]
            self._fn_prev = fn
        else:
            fn, ft, fres = hc.fn, 0.0, hc.fn
            fhc, closing = hc.fn, hc.ddelta
            pen, k_eq, tc, ncon = max(hc.delta, 0.0), hc.k_eq, 0.0, int(hc.fn > 1e-3)
            tau_ext = d.qfrc_applied[:m.nv]
        L = self.log
        L["t"].append(t); L["fn"].append(fn); L["ft"].append(ft); L["fresultant"].append(fres)
        L["fmeas"].append(F_meas); L["fn_lpf"].append(self.native_plot_filter.read(fn))
        L["fhc"].append(fhc); L["fhc_lpf"].append(self.hc_plot_filter.read(fhc))
        L["delta"].append(pen); L["ddelta"].append(closing); L["k"].append(k_eq)
        L["compression"].append(compression)
        L["solref_tc"].append(tc); L["ncontact"].append(ncon)
        L["zs"].append(zs); L["ztip"].append(pos[2]); L["x"].append(pos[0])
        L["tau"].append(np.abs(d.ctrl[:]).max())
        L["fext_tau"].append(float(np.linalg.lstsq(self._J_pos.T, tau_ext, rcond=None)[0] @ n))
        L["fcmd_z"].append(0.0)
        probe_center = d.geom_xpos[ids["tip_geom"]]
        self.visual_deformation = (
            (float(probe_center[0]), float(probe_center[1]), compression) if compression > 0 else None
        )
        for k, val in (("Kd", K_d), ("Bd", B_d), ("Kt", K_t), ("v_in", v_in), ("dist", dist),
                       ("Fdes_eff", F_des_eff), ("F_virt", F_virt), ("v_lim", v_lim), ("y", pos[1]),
                       ("x_ref", goal[0]), ("y_ref", goal[1]), ("z_ref", goal[2]), ("scan_s", scan_s),
                       ("exy", np.hypot(*e_xy[:2]))):
            L[k].append(float(val))
        return fn
