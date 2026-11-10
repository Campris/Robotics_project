"""
Geometria, respiro, eventi e modello di contatto Hunt-Crossley del torso.

La superficie e' analitica z = zs(x, y, t) (geometria + respiro + eventi).
Tissue.contact() calcola la forza del tessuto sulla sonda (Hunt-Crossley): e' QUESTA la forza
che il controllore di ammettenza applica al robot (panda_scene.apply_tip_force).

Convenzioni: unita' SI. z verso l'alto, la normale n punta fuori dal corpo.
delta > 0 = penetrazione della punta nel tessuto.
"""
from dataclasses import dataclass
from collections import deque
from typing import Callable, Optional
import numpy as np


def sigmoid(x):                                              #sigmoid((u-c/w)) dove c è il centro della transizione e w la larghezza della transizione
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def min_jerk(s):
    """0 -> 1 con derivata prima e seconda nulle agli estremi (C2)."""
    s = np.clip(s, 0.0, 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def min_jerk_dot(s):
    """Derivata di min_jerk rispetto a s."""
    s = np.clip(s, 0.0, 1.0)
    return 30 * s**2 - 60 * s**3 + 30 * s**4

#---------------------------------------------------------------------------
# Geometria
# --------------------------------------------------------------------------
@dataclass
class TorsoGeometry:
    """Sezione ellittica (curvatura trasversale) con asse x cranio-caudale."""
    H: float = 0.10        # altezza dell'apice [m]
    B: float = 0.20        # semiasse trasversale [m]
    taper: float = 0.0     # variazione di altezza lungo x
    z_base: float = 0.0

    def z0(self, x, y):
        r = np.clip(y / self.B, -0.9, 0.9)
        return self.z_base + self.H * np.sqrt(1.0 - r**2) * (1.0 + self.taper * x)


# --------------------------------------------------------------------------
# Rigidita' variabile con la posizione
# --------------------------------------------------------------------------
@dataclass
class StiffnessMap:
    """
    Rigidita' EQUIVALENTE lineare k_eq(x, y) [N/m] alla profondita' di riferimento
    (forza / penetrazione a delta_ref). Torace (x < x_tr): piu' rigido, con costole
    e sterno. Addome (x > x_tr): morbido. Tutto liscio, senza gradini.
    """
    k_abdomen: float = 500.0
    k_chest: float = 900.0
    k_rib: float = 600.0
    k_sternum: float = 700.0
    x_tr: float = 0.02
    rib_pitch: float = 0.03

    def __call__(self, x, y):
        w_th = 1.0 - sigmoid((x - self.x_tr) / 0.02)          #sigmoide che va circa 1 a 4 larghezze della transizione tra torace e addome
        ribs = (0.5 * (1 + np.cos(2 * np.pi * x / self.rib_pitch))) ** 2    #(0.5·(1 + cos(2π·u/T)))²	
        ribs *= sigmoid((abs(y) - 0.03) / 0.01)         
        sternum = np.exp(-0.5 * (y / 0.015) ** 2)     #exp(−0.5·(u/σ)²)  campana gaussiana
        return (self.k_abdomen
                + w_th * ((self.k_chest - self.k_abdomen)
                          + self.k_rib * ribs + self.k_sternum * sternum))


# --------------------------------------------------------------------------
# Respiro ed eventi
# --------------------------------------------------------------------------
@dataclass
class Breathing:
    """
    u(x, t) = gain(x) * s(t) [m]: gain = QUANTO sale quel punto, s = QUANDO (0 espirato, 1 inspirato, max 1.15).
    Con fn = la TUA funzione t -> s si sostituisce il modello di s() (period viene ignorato).
    """
    period: float = 4.0                            # [s] durata di un respiro (QUANDO)
    amp_chest: float = 0.0015                      # [m] salita del torace a s = 1 (QUANTO)
    amp_abdomen: float = 0.0055                    # [m] salita dell'addome a s = 1 (QUANTO)
    x_tr: float = 0.02                             # [m] centro transizione torace-addome (NON e' StiffnessMap.x_tr)
    fn: Optional[Callable[[float], float]] = None  # funzione t -> s; None = modello qui sotto

    def s(self, t):
        if self.fn is not None:
            return self.fn(t)                               # usa la funzione passata ed esce
        phi = 2 * np.pi * t / self.period                   # tempo -> angolo: un respiro = un giro
        phi_w = phi + 0.35 * np.sin(phi)                    # QUANDO: forma del ciclo (resta simmetrica, cima a T/2)
        mod = 1.0 + 0.15 * np.sin(2 * np.pi * t / 17.0)     # QUANTO: profondita' 0.85..1.15, periodo 17 s
        return mod * 0.5 * (1.0 - np.cos(phi_w))            # gobba 0 -> 1 -> 0 (parte espirato), scalata da mod

    def gain(self, x):
        return self.amp_chest + (self.amp_abdomen - self.amp_chest) * sigmoid((x - self.x_tr) / 0.03)  # media in x_tr

    def u(self, x, t):
        return self.gain(x) * self.s(t)


@dataclass
class Event:
    """Evento liscio (C2). 'bump' sale e scende (tosse, singhiozzo); 'step' resta (spostamento)."""
    t0: float               #siccome non c'è l' = non sono di default, ma vanno settati ogni volta che si crea un oggetto Event
    duration: float
    amplitude: float       # [m], positivo = la pelle sale verso la sonda
    kind: str = "bump"

    def value(self, t):
        s = (t - self.t0) / self.duration   #tempo locale dell'evento, normalizzato alla durata
        if self.kind == "step":             #evento del tipo: pelle sale e resta alta
            return self.amplitude * min_jerk(s)   
        if s <= 0.0 or s >= 1.0:            #qui sono fuori dall'evento, quindi non c'è spostamento della pelle
            return 0.0
        return self.amplitude * min_jerk(1.0 - abs(2.0 * s - 1.0))  #sto facendo smussamento di segnale triangolare 1-abs(2t-1)
                                                                    #che va da 0 a 1 a 0, e lo smusso con min_jerk per avere continuita' C2


# --------------------------------------------------------------------------
# Tessuto + contatto Hunt-Crossley
# --------------------------------------------------------------------------
@dataclass
class ContactResult:
    force: np.ndarray      # forza del tessuto sulla sonda [N], mondo
    fn: float              # modulo della forza normale [N]
    delta: float           # penetrazione [m]
    ddelta: float          # velocita' di penetrazione [m/s]
    k_eq: float            # rigidita' equivalente locale [N/m]
    normal: np.ndarray     # normale uscente della pelle sotto la punta
    zs: float              # quota della superficie sotto la punta [m]


class Tissue:
    """
    Contatto Hunt-Crossley: F = k * d^n * (1 + 1.5 * alpha * d_dot)
      k      [N/m^n]  derivato da k_eq(x,y): k = k_eq * delta_ref^(1-n)
      n      1.5 = Hertz (rigidita' che cresce con la penetrazione)
      alpha  [s/m]  dissipazione
    Anti-spike:
      * penetrazione liscia (softplus, scala eps) -> nessun gradino all'inizio del contatto
      * il fattore di smorzamento e' saturato a >= 0 -> la forza non diventa mai negativa
      * il termine viscoso e' proporzionale a d^n -> va a zero con continuita' al distacco
      * respiro/eventi C2 -> accelerazione limitata della superficie
      * nel calcolo di d_dot entra anche lo scorrimento lungo la superficie curva
    """

    def __init__(self, geom=None, stiff=None, breathing=None, events=None,
                 n=1.5, alpha=1.5, delta_ref=0.008, eps=1e-4):
        self.geom = geom or TorsoGeometry()                      #questo or è per dire se nessuno me l'ha dato uso il default
        self.stiff = stiff or StiffnessMap()                                       #mappa della rigidezza, default = StiffnessMap() con parametri di default
        self.breathing = breathing or Breathing()                                  #mod ello di respiro, default = Breathing() con parametri di default
        self.events = list(events or [])                                           # lista di eventi (vuota di default)
        self.n, self.alpha, self.delta_ref, self.eps = n, alpha, delta_ref, eps    # parametri del contatto (A6)

    # -- superficie --------------------------------------------------------
    def surface_height(self, x, y, t):
        z = self.geom.z0(x, y) + self.breathing.u(x, t)       #forma a riposo + respiro
        for e in self.events:                                 #per ogni evento, se attivo, sommo lo spostamento della pelle dovuto all'evento
            z += e.value(t)
        return z

    def _grad(self, x, y, t, h=1e-4):
        gx = (self.surface_height(x + h, y, t) - self.surface_height(x - h, y, t)) / (2 * h)  #derivata parziale della superficie rispetto a x
        gy = (self.surface_height(x, y + h, t) - self.surface_height(x, y - h, t)) / (2 * h)   #derivata parziale della superficie rispetto a y
        return gx, gy

    def normal(self, x, y, t):
        gx, gy = self._grad(x, y, t)
        n = np.array([-gx, -gy, 1.0])
        return n / np.linalg.norm(n)

    # -- contatto ----------------------------------------------------------
    def contact(self, p, v, t, h=5e-4) -> ContactResult:
        x, y, zp = p                                # punta VERA (da MuJoCo): x, y e quota zp
        zs = self.surface_height(x, y, t)           # quota della pelle sotto la punta, adesso
        gx, gy = self._grad(x, y, t)                # pendenze della pelle in x e y
        nvec = np.array([-gx, -gy, 1.0])            # normale uscente (non ancora lunga 1)
        inv_norm = 1.0 / np.linalg.norm(nvec)
        n_hat = nvec * inv_norm                     # normale lunga 1
        nz = n_hat[2]                               # = cos(inclinazione della pelle)

        delta_raw = (zs - zp) * nz                  # penetrazione lungo n: >0 punta dentro, <0 in aria
        zs_dot = (self.surface_height(x, y, t + h) - self.surface_height(x, y, t - h)) / (2 * h)  # pelle sale/scende (respiro, tosse)
        ddelta_raw = (zs_dot + gx * v[0] + gy * v[1] - v[2]) * nz  # vel. penetrazione (>0 entra) = pelle si muove + punta scivola di lato - punta scende

        # penetrazione liscia (softplus) e sua derivata
        d = self.eps * np.logaddexp(0.0, delta_raw / self.eps)   # come max(0, delta_raw) ma senza spigolo (eps = 0.1 mm)
        dd = sigmoid(delta_raw / self.eps) * ddelta_raw          # velocita' che conta solo a contatto (fuori = 0)

        k_eq = float(self.stiff(x, y))                           # rigidezza della mappa (A3) sotto la punta
        k = k_eq * self.delta_ref ** (1.0 - self.n)              # k di Hunt-Crossley: a delta_ref da' k_eq * delta_ref
        damping_factor = max(0.0, 1.0 + 1.5 * self.alpha * dd)   # >1 entra, <1 esce; mai <0 (niente forza che tira)
        fn = k * d**self.n * damping_factor                      # Hunt-Crossley: modulo della forza [N]
        return ContactResult(fn * n_hat, fn, delta_raw, ddelta_raw, k_eq, n_hat, zs)  # force = fn lungo la normale


# --------------------------------------------------------------------------
# Sensore di forza "franka" (modello della stima F_ext del Panda), alternativo all'ATI
# --------------------------------------------------------------------------
class ForceSensor:
    """LPF del primo ordine + ritardo + bias + rumore. Parametri da calibrare sul robot reale."""

    def __init__(self, dt, fc=80.0, noise_std=0.15, bias=0.0, delay_s=0.003, seed=0):
        self.a = 1.0 - np.exp(-2 * np.pi * fc * dt)
        self.y = 0.0
        self.noise_std, self.bias = noise_std, bias
        self.buf = deque([0.0] * (int(round(delay_s / dt)) + 1), maxlen=int(round(delay_s / dt)) + 1)
        self.rng = np.random.default_rng(seed)

    def read(self, f_true):
        self.y += self.a * (f_true - self.y)
        self.buf.append(self.y)
        return self.buf[0] + self.bias + self.rng.normal(0.0, self.noise_std)
