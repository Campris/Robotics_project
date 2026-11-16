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
# =====================================================================================================
#  COME SONO SCRITTI I COMMENTI DI STUDIO
#    COSA   = cosa fa la riga, pezzo per pezzo (sintassi, funzione, matematica)
#    PERCHÉ = a cosa serve qui e cosa succederebbe senza
#    DOVE   = chi usa il risultato nella simulazione (file -> funzione -> blocco)
#  I numeri tra parentesi sono MISURATI eseguendo il codice (simulazione di default: serpentina che parte da
#  (-0.16, -0.11), sussulti con seme 0; 13-11-2026).
#
#  MAPPA DEL FILE (chi usa cosa)
#    AdmittanceParams              tutti i numeri del controllore     <- run_admittance.make_params, AdmittanceSim
#    StiffnessSchedule   (BL. 1)   K_d, rho, v_in_max                 <- step(): BLOCCHI 1 e 3
#    tissue_stiffness,
#    critical_damping    (BL. 2)   K_t e B_d                          <- step(): BLOCCO 2
#    PureIntegratorAdmittance (3)  l'equazione di ammettenza          <- step(): BLOCCO 3
#    LineTrajectory      (BL. 4a)  serpentina in XY                   <- step(): BLOCCO 4a
#    differential_ik     (BL. 5)   da v6 a q_dot                      <- step(): BLOCCO 5
#    AdmittanceSim                 monta tutto; step() = 1 ms         <- run_admittance (ciclo batch o viewer)
#  ORDINE DI ESECUZIONE dentro step(): 0 -> 1 -> 2 -> 4a -> 3 -> 4 -> 5 -> 6
#  (il 4a viene PRIMA del 3 perche' il 3 ha bisogno dell'obiettivo 'goal' calcolato nel 4a).
#
#  TIMELINE MISURATA (default): contatto 3.333 s, K_d = 0 da 3.339 s, F_meas > 2 N a 3.615 s, la serpentina
#  parte a 5.615 s e finisce a ~170 s; a regime F_meas 4.994 N di media, errore RMS 0.051 N.
# =====================================================================================================
# COSA: @dataclass genera da solo il costruttore di una classe di soli dati (AdmittanceParams sotto).
from dataclasses import dataclass
import numpy as np
import mujoco

# COSA: importa gli altri file del progetto. ps = panda_scene (modello, jac, rot_err, apply_tip_force);
#       Tissue = pelle + contatto Hunt-Crossley; min_jerk / min_jerk_dot = rampa liscia 0 -> 1 e sua derivata;
#       ATINanoSensor = simulazione del sensore ATI.
import panda_scene as ps
from tissue import Tissue, min_jerk, min_jerk_dot
from ft_sensor import ATINanoSensor


# =====================================================================================================
#  PARAMETRI  (tutto quello che puoi manipolare e' qui)
# =====================================================================================================
# COSA: @dataclass trasforma ogni riga "NOME: tipo = valore" in un argomento del costruttore con quel default:
#       AdmittanceParams() ha tutti i default, AdmittanceParams(F_DES=8.0) cambia solo F_DES.
#       I default sono tuple (immutabili): una lista come default sarebbe vietata da dataclass.
# PERCHÉ: un solo posto con TUTTI i numeri del controllore.
# DOVE: AdmittanceSim.self.p (letto in tutti i blocchi); run_admittance.make_params lo modifica con setattr
#       secondo le opzioni da riga di comando.
@dataclass
class AdmittanceParams:
    # --- forza obiettivo --------------------------------------------------------------------------
    # COSA: forza di contatto desiderata lungo la normale della pelle [N].
    # DOVE: BLOCCO 1 (F_des_eff = rho * F_DES), tissue_stiffness in modo "tangent", run_admittance (statistiche).
    F_DES: float = 5.0             # forza di contatto desiderata [N]

    # --- BLOCCO 1: soglie di K_d (distanza d dalla pelle, in metri, positiva = sopra la pelle) ------
    # COSA: rigidezza della molla virtuale che tira la punta verso l'obiettivo del planner (F_virt = K_d * e_in).
    # PERCHÉ: con l'ammettenza, a regime B_d * v = K_d * e  ->  v = (K_d / B_d) * e: K_d / B_d fa da guadagno
    #         di posizione [1/s]. Lontano: 1500 / 13.7 ~ 110 1/s (insegue la discesa da vicino, come un robot
    #         rigido e preciso). Vicino: 300 / 13.7 ~ 22 1/s (piu' morbido prima di toccare).
    # DOVE: StiffnessSchedule.K (BLOCCO 1) -> F_virt (BLOCCO 3).
    K_FAR: float = 1500.0          # rigidezza virtuale lontano dalla pelle [N/m]
    K_NEAR: float = 300.0          # rigidezza virtuale alla soglia D_NEAR [N/m]
    # COSA: le tre soglie di distanza (d > 0 = sopra la pelle). Misurato, K_d senza forza: d = 60 mm -> 1500,
    #       40 mm -> 900, 20 mm -> 300, 10 mm -> 150, 5 mm -> 31, 0 -> 0.
    # PERCHÉ: HOVER = 0.06 = D_FAR: la discesa parte esattamente dove K_d comincia a calare.
    # DOVE: StiffnessSchedule.s_a / s_b.
    D_FAR: float = 0.060           # d >= D_FAR          -> K_d = K_FAR
    D_NEAR: float = 0.020          # D_NEAR <= d < D_FAR -> K_d scende da K_FAR a K_NEAR
    D_OFF: float = 0.000           # D_OFF <= d < D_NEAR -> K_d scende da K_NEAR a 0; d <= D_OFF -> K_d = 0
    # COSA: soglie di FORZA misurata: sotto 0.5 N il "cancello" e' aperto (1), sopra 2 N chiuso (0); in mezzo
    #       rampa min-jerk (misurato: 1.0 N -> 0.79, 1.25 N -> 0.5, 1.5 N -> 0.21).
    # PERCHÉ: d viene dal modello della pelle; se la stima fosse sbagliata (pelle spostata, modello diverso dal
    #         vero), la forza dice comunque "sto toccando" e spegne K_d. F_GATE1 decide anche quando parte il
    #         clock della traiettoria e quando il limite di velocita' diventa V_CONTACT.
    # DOVE: StiffnessSchedule.force_gate e v_in_max; LineTrajectory.update (t_contact).
    F_GATE0: float = 0.5           # rete di sicurezza in forza: sopra F_GATE0 K_d inizia a spegnersi...
    F_GATE1: float = 2.0           # ...e sopra F_GATE1 e' completamente spento (anche se la stima di d e' sbagliata)

    # --- BLOCCO 2: smorzamento --------------------------------------------------------------------
    # COSA: massa virtuale M_d dell'equazione di ammettenza [kg].
    # PERCHÉ: a contatto il sistema "ammettenza + tessuto" e' massa-molla-smorzatore con molla K_t: pulsazione
    #         propria sqrt(K_t / M_d) = sqrt(937 / 0.05) ~ 137 rad/s al punto di partenza. M_d piccola = risposta
    #         veloce, ma troppo piccola rende il sistema sensibile a rumore e ritardi.
    # DOVE: PureIntegratorAdmittance (BLOCCO 3) e critical_damping (BLOCCO 2).
    M_D: float = 0.05              # massa virtuale [kg]  (50 g: ammettenza molto reattiva)
    # COSA: come scegliere K_t, la rigidezza del tessuto usata per lo smorzamento (vedi tissue_stiffness).
    KT_MODE: str = "k_eq"          # "k_eq": K_t = k_eq(x,y) della tua mappa (come nella proposta)
                                   # "tangent": pendenza dF/d(delta) del modello Hunt-Crossley al punto di lavoro F_DES
                                   #            (circa 0.9 k_eq a 5 N: punto di lavoro sotto delta_ref, vedi tissue_stiffness)
    # COSA: se True somma K_d a K_t nello smorzamento. Default False: a contatto K_d = 0 e non cambia niente;
    #       cambia solo in avvicinamento.  DOVE: BLOCCO 2.
    B_INCLUDE_KD: bool = False     # False: B_d = 2 sqrt(M K_t) (proposta) | True: 2 sqrt(M (K_t + K_d)) anche in avvicinamento
    # COSA: valore minimo di B_d. PERCHÉ: se K_t fosse 0 (o quasi) B_d = 0 e l'ammettenza non avrebbe attrito.
    B_MIN: float = 1.0             # [Ns/m] limite inferiore di sicurezza

    # --- BLOCCO 3: integratore ---------------------------------------------------------------------
    # COSA: schema numerico per integrare l'equazione (vedi PureIntegratorAdmittance.step).
    INTEGRATOR: str = "implicit"   # "explicit" | "implicit" (Eulero esplicito o implicito sul termine di smorzamento)

    # --- limiti di velocita' (lungo la normale) ----------------------------------------------------
    # COSA: limiti di v_in [m/s]. Lontano fino a 10 cm/s; entro D_NEAR 2 cm/s; a contatto 2 cm/s; in uscita 10 cm/s.
    # PERCHÉ: toccare il paziente piano (urto piccolo) anche se la molla virtuale "tirerebbe" di piu'
    #         (misurato: a t = 2.0 s l'ammettenza chiede 21.4 mm/s e il limite e' proprio 21.4).
    # DOVE: StiffnessSchedule.v_in_max (limite superiore) e BLOCCO 3 (clamp fra -V_RETRACT e il limite).
    V_CRUISE: float = 0.10         # velocita' massima di avvicinamento lontano dalla pelle [m/s]
    V_TOUCH: float = 0.02          # velocita' massima di avvicinamento entro D_NEAR [m/s]
    V_CONTACT: float = 0.02        # velocita' massima verso la pelle quando c'e' contatto [m/s]
    V_RETRACT: float = 0.10        # velocita' massima di allontanamento [m/s]

    # --- BLOCCO 4a: traiettoria in XY --------------------------------------------------------------
    # COSA: accende la scansione in XY (False = solo discesa e mantenimento della forza nel primo punto del percorso).
    SCAN_ENABLE: bool = True
    # COSA: forma del percorso: "serpentine" (righe avanti e indietro) o "line" (un solo segmento).
    SCAN_PATTERN: str = "serpentine"    # "line" mantiene il percorso singolo definito da SCAN_START/END
    # COSA: rettangolo coperto dalla serpentina: x da -16 a +16 cm, y da -11 a +11 cm (la patch e' +-20 x +-15).
    SCAN_X_RANGE: tuple = (-0.16, 0.16) # [m] limiti cranio-caudali, con margine dal bordo
    SCAN_Y_RANGE: tuple = (-0.11, 0.11) # [m] limiti trasversali, con margine dal bordo
    # COSA: distanza massima fra due righe. Misurato: 7 righe a 36.7 mm (<= 40 mm), 14 waypoint, 2.46 m di percorso.
    SCAN_ROW_SPACING: float = 0.04      # [m] distanza massima tra le passate
    # COSA: estremi del segmento usato solo con SCAN_PATTERN = "line".
    SCAN_START: tuple = (0.04, 0.06)    # [m] inizio/fine della linea singola
    SCAN_END: tuple = (-0.08, 0.06)    # [m] fine/inizio della linea singola
    # COSA: velocita' MEDIA lungo il percorso: 2.46 m / 0.015 = 164 s. Col profilo min-jerk la velocita' di picco
    #       a meta' segmento e' 1.875 volte la media: 28 mm/s sulle righe lunghe.
    SCAN_SPEED: float = 0.015           # [m/s] velocita' MEDIA sul percorso, con profilo min-jerk per segmento
    # COSA: attesa dopo il primo contatto (F_meas > F_GATE1) prima di muoversi. PERCHÉ: lascia assestare la forza.
    SCAN_DELAY: float = 2.0             # [s] attesa dopo il primo contatto prima di partire
    # COSA: soglia sotto cui la traiettoria si ferma. PERCHÉ: non trascinare la sonda in aria se si perde il contatto.
    F_CONTACT_OK: float = 1.0           # [N] il clock della traiettoria avanza solo se F_meas supera questa soglia

    # --- BLOCCO 4: assi non controllati in forza ---------------------------------------------------
    # COSA: guadagno P sull'errore in XY: v = 4 * errore -> costante di tempo 1/4 = 0.25 s.
    #       Misurato in scansione: errore XY medio 0.49 mm, massimo 2.1 mm.
    KP_POS_V: float = 4.0          # [1/s] guadagno P posizione tangenziale -> velocita'
    # COSA: guadagno P sull'errore di orientamento (rot_err) -> velocita' angolare.
    KP_ROT_V: float = 4.0          # [1/s] guadagno P orientamento -> velocita' angolare
    # COSA: saturazioni: velocita' tangente max 10 cm/s, velocita' angolare max 0.5 rad/s per componente.
    V_TAN_MAX: float = 0.10        # [m/s]
    W_MAX: float = 0.5             # [rad/s]

    # --- BLOCCO 5: IK differenziale ----------------------------------------------------------------
    # COSA: lambda della pseudo-inversa smorzata: nella formula entra lambda^2 = 4e-4.
    IK_DAMPING: float = 0.02       # lambda della pseudo-inversa smorzata
    # COSA: guadagno del richiamo verso la postura q_null nel nullspace.
    K_NULL_V: float = 1.0          # [1/s] richiamo alla postura nel nullspace
    # COSA: velocita' massima di ogni giunto.
    QDOT_MAX: float = 1.0          # [rad/s] limite sulla velocita' di ogni giunto
    # COSA: distanza massima fra riferimento q_ref e giunto vero q. Misurato in contatto: |q_ref - q| <= 0.0036 rad.
    Q_REF_MAX_ERR: float = 0.08    # [rad] anti-windup: |q_ref - q| limitato

    # --- BLOCCO 6: controllore di posizione di giunto ---------------------------------------------
    # COSA: rigidezze del PD di giunto [Nm/rad]: 600 per spalla/gomito, piu' basse sul polso (giunti leggeri).
    KP_JOINT: tuple = (600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0)   # [Nm/rad] (valori Franka)
    # COSA: fattore di smorzamento del PD di giunto: 1 = critico (vedi BLOCCO 6).
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

    # COSA: memorizza i parametri; la classe non ha altro stato (le sue funzioni dipendono solo da d e F_meas).
    # DOVE: creato in AdmittanceSim.__init__ (self.schedule) e in run_admittance.plot_schedule.
    def __init__(self, p: AdmittanceParams):
        self.p = p

    # COSA: (d - D_NEAR) / (D_FAR - D_NEAR) porta l'intervallo [20 mm, 60 mm] in [0, 1]; min_jerk lo taglia fuori
    #       da [0, 1] e lo rende liscio: 10 s^3 - 15 s^4 + 6 s^5. Quindi s_a = 0 sotto 20 mm, 1 sopra 60 mm,
    #       0.5 a 40 mm. float() toglie il tipo numpy.
    # PERCHÉ: e' la "fase a)": dice quanto di K_FAR resta. Liscia (C2) -> niente scalini di forza virtuale.
    # DOVE: K() e v_in_max().
    def s_a(self, d):
        return float(min_jerk((d - self.p.D_NEAR) / (self.p.D_FAR - self.p.D_NEAR)))

    # COSA: stessa cosa su [D_OFF, D_NEAR] = [0, 20 mm]: 0 sulla pelle o dentro, 1 da 20 mm in su, 0.5 a 10 mm.
    # PERCHÉ: e' la "fase b)": spegne del tutto K_d arrivando sulla pelle.  DOVE: K() e rho().
    def s_b(self, d):
        return float(min_jerk((d - self.p.D_OFF) / (self.p.D_NEAR - self.p.D_OFF)))

    def force_gate(self, F_meas):
        """1 senza contatto, 0 con contatto stabilito (forza misurata > F_GATE1)."""
        p = self.p
        # COSA: min_jerk((F - 0.5) / 1.5) va da 0 (F <= 0.5 N) a 1 (F >= 2 N); "1 - ..." lo capovolge:
        #       cancello aperto = 1 senza forza, chiuso = 0 con contatto.
        # DOVE: K() (moltiplica K_d) e rho() (accende F_des).
        return 1.0 - float(min_jerk((F_meas - p.F_GATE0) / (p.F_GATE1 - p.F_GATE0)))

    def K(self, d, F_meas=0.0):
        p = self.p
        # COSA: parte di distanza: [K_NEAR + (K_FAR - K_NEAR) * s_a] va da 300 a 1500; moltiplicata per s_b
        #       va a 0 sulla pelle. Esempio: d = 40 mm -> (300 + 1200 * 0.5) * 1 = 900 N/m.
        k_dist = (p.K_NEAR + (p.K_FAR - p.K_NEAR) * self.s_a(d)) * self.s_b(d)
        # COSA: moltiplica per il cancello in forza: basta UNO dei due (distanza o forza) per spegnere K_d.
        # DOVE: step() BLOCCO 1: K_d -> F_virt = K_d * e_in (BLOCCO 3); log "Kd".
        #       Misurato: 1487 N/m a t = 0.5 s (d = 56 mm), 322 a 2.0 s (25 mm), 94 a 2.9 s (8 mm), 0 da 3.339 s.
        return k_dist * self.force_gate(F_meas)

    def rho(self, d, F_meas=0.0):
        """Quanto della forza desiderata e' attivo (0 lontano, 1 a contatto): subentra mentre K_d si spegne."""
        # COSA: 1 - s_b(d) vale 0 sopra 20 mm e 1 sulla pelle; 1 - force_gate vale 0 senza forza e 1 sopra 2 N.
        #       max(...) prende il piu' grande: basta uno dei due segnali di "contatto".
        # PERCHÉ: passaggio di consegne: mentre la molla virtuale (K_d) si spegne, la forza desiderata si accende.
        #         Senza rho, lontano dalla pelle e_F = 5 - 0 = 5 N spingerebbe la sonda verso il basso SEMPRE.
        # DOVE: step() BLOCCO 1: F_des_eff = rho * F_DES. Misurato: 0 a 25 mm, 0.685 a 8 mm, 1 a contatto.
        return max(1.0 - self.s_b(d), 1.0 - self.force_gate(F_meas))

    def v_in_max(self, d, F_meas=0.0):
        """Velocita' massima verso la pelle: scende a V_TOUCH vicino alla pelle; con contatto vale V_CONTACT."""
        p = self.p
        # COSA: se la forza misurata supera 2 N siamo in contatto: limite V_CONTACT (2 cm/s), ed esce.
        if F_meas > p.F_GATE1:
            return p.V_CONTACT
        # COSA: altrimenti interpolazione liscia fra V_TOUCH (s_a = 0, sotto 20 mm) e V_CRUISE (s_a = 1, sopra 60 mm).
        #       Misurato: 60 mm/s a d = 40 mm, 28 mm/s a 30 mm, 20 mm/s sotto 20 mm.
        # DOVE: step() BLOCCO 3: v_lim -> clamp della velocita' dell'ammettenza.
        return p.V_TOUCH + (p.V_CRUISE - p.V_TOUCH) * self.s_a(d)


# =====================================================================================================
#  BLOCCO 2: SMORZAMENTO CRITICO DEL TESSUTO
# =====================================================================================================
def tissue_stiffness(tissue, x, y, F_des, mode="k_eq"):
    """
    K_t [N/m] del tessuto nel punto (x, y).
      "k_eq"    : valore della mappa (pendenza dF/d(delta) a delta_ref = 8 mm)
      "tangent" : pendenza dF/d(delta) del modello Hunt-Crossley F = k d^n al punto di lavoro F = F_des.
                  d* = (F_des / k)^(1/n)  ->  K_t = n * F_des / d*      (= k_eq * sqrt(d*/delta_ref): circa 0.9 k_eq a 5 N)
    """
    # COSA: tissue.stiff e' la StiffnessMap: chiamarla come funzione esegue il suo __call__(x, y) -> k_eq [N/m].
    #       Misurato lungo la serpentina: da 500 (addome) a 1629 N/m (torace con costole e sterno).
    k_eq = float(tissue.stiff(x, y))
    # COSA: modo di default: K_t = valore della mappa, e la funzione finisce qui.
    if mode == "k_eq":
        return k_eq
    # COSA (modo "tangent"): esponente di Hertz n = 1.5.
    n = tissue.n
    # COSA: la stessa k di Hunt-Crossley usata in Tissue.contact: k = k_eq / (n * delta_ref^(n-1)),
    #       scelta in modo che a delta = delta_ref (8 mm) la pendenza dF/d(delta) valga proprio k_eq.
    k = k_eq / (n * tissue.delta_ref ** (n - 1.0))   # stessa k del tessuto (tissue.py, contact)
    # COSA: penetrazione statica d* alla quale F = k d*^n = F_des: invertendo, d* = (F_des / k)^(1/n).
    d_star = (F_des / k) ** (1.0 / n)
    # COSA: pendenza al punto di lavoro: F = k d^n -> dF/dd = n k d^(n-1) = n F / d  ->  n F_des / d*.
    #       Misurato: (0, 0) 1181 contro k_eq 1325 (0.89); addome (0.1, 0) 633 contro 520 (1.22): dove il tessuto
    #       e' morbido d* supera 8 mm e la pendenza vera e' piu' alta di k_eq.
    # DOVE: step() BLOCCO 2 -> critical_damping -> B_d; log "Kt".
    return n * F_des / d_star


def critical_damping(M_d, K, B_min=0.0):
    """B_d = 2 sqrt(M_d K): smorzamento critico di un sistema massa-molla M ddx + B dx + K x = 0."""
    # COSA: B_d = 2 sqrt(M_d K), ma mai sotto B_min.
    # PERCHÉ (la matematica): a contatto K_d = 0 e l'ammettenza e' M_d v' + B_d v = F_des - F_meas. Se la punta
    #       affonda di x, il tessuto risponde con F_meas ~ F_0 + K_t x, e v = x' (velocita' verso la pelle):
    #           M_d x'' + B_d x' + K_t x = costante   ->   massa-molla-smorzatore, con la molla = il TESSUTO.
    #       Smorzamento relativo zeta = B_d / (2 sqrt(M_d K_t)): con B_d = 2 sqrt(M_d K_t) zeta = 1 (critico) =
    #       la risposta piu' rapida SENZA sovraelongazione (niente picchi di forza sopra F_des).
    #       Siccome K_t cambia col punto (addome/torace), B_d viene ricalcolato a ogni ms.
    #       Misurato: B_d da 10 (addome, 500 N/m) a 18 Ns/m (sterno/costole), media in scansione 12.7.
    # DOVE: step() BLOCCO 2 -> B_d -> adm.step(F, B_d) (BLOCCO 3); log "Bd".
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
    # NB (misurato): la docstring qui sopra e' vecchia: con M_D = 0.05 la costante M/B e' 2.8-4.9 ms (non 20).
    # Ogni passo il fattore che moltiplica v vale 1 - dt*B/M ~ 0.73 (esplicito) contro 1/(1 + dt*B/M) ~ 0.79
    # (implicito): entrambi stabili, ma non "uguali".

    # COSA: memorizza massa virtuale, passo (1 ms) e schema; lo STATO dell'ammettenza e' solo self.v,
    #       che parte da 0 (robot fermo).
    # DOVE: creato in AdmittanceSim.__init__ (self.adm); self.v serve anche per riprendere la simulazione.
    def __init__(self, M_d, dt, integrator="implicit"):
        self.M, self.dt, self.integrator = M_d, dt, integrator
        self.v = 0.0

    def step(self, F_net, B_d):
        if self.integrator == "explicit":
            # COSA: Eulero ESPLICITO: dv/dt = (F_net - B v) / M valutata con la v VECCHIA:
            #       v[k+1] = v[k] + dt/M * (F_net - B v[k]).
            #       Stabile solo se dt*B/M < 2, cioe' B < 2M/dt = 100 Ns/m (noi al massimo 18: OK).
            self.v = self.v + self.dt / self.M * (F_net - B_d * self.v)
        else:
            # COSA: Eulero IMPLICITO: lo smorzamento valutato con la v NUOVA:
            #       M (v1 - v0)/dt = F_net - B v1  ->  v1 (M + dt B) = M v0 + dt F_net  ->  v1 = (M v0 + dt F_net)/(M + dt B).
            #       Stabile per qualunque B >= 0 (il fattore 1/(1 + dt B/M) e' sempre fra 0 e 1).
            self.v = (self.M * self.v + self.dt * F_net) / (self.M + self.dt * B_d)
        # DOVE: step() BLOCCO 3: v_in = self.adm.step(F_virt + e_F, B_d), subito dopo limitata da clamp.
        return self.v

    def clamp(self, v_min, v_max):
        """Limita la velocita' e RIPORTA nello stato il valore limitato (anti-windup per back-calculation)."""
        # COSA: np.clip taglia v in [v_min, v_max] e il risultato SOSTITUISCE lo stato self.v.
        # PERCHÉ: se tagliassi solo l'uscita, lo stato continuerebbe a crescere oltre il limite (windup): in discesa
        #         la molla chiede piu' di 20 mm/s per secondi; a contatto la v "accumulata" dovrebbe prima
        #         scaricarsi e la sonda affonderebbe troppo. Riscrivendo lo stato, il passo dopo parte dal valore
        #         che il robot sta davvero eseguendo.
        # DOVE: step() BLOCCO 3: v_in = self.adm.clamp(-V_RETRACT, v_lim).
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
        # COSA: controlli sugli ingressi: con velocita' o passo <= 0 le divisioni sotto non avrebbero senso.
        if p.SCAN_SPEED <= 0 or p.SCAN_ROW_SPACING <= 0:
            raise ValueError("SCAN_SPEED e SCAN_ROW_SPACING devono essere positivi")
        # COSA: percorso "line": due soli waypoint, inizio e fine; array numpy 2 x 2 di float.
        if p.SCAN_PATTERN == "line":
            self.waypoints = np.array([p.SCAN_START, p.SCAN_END], dtype=float)
        # COSA: percorso a serpentina.
        elif p.SCAN_PATTERN == "serpentine":
            # COSA: spacchetta le tuple dei limiti: x_min = -0.16, x_max = 0.16, y_min = -0.11, y_max = 0.11.
            x_min, x_max = p.SCAN_X_RANGE
            y_min, y_max = p.SCAN_Y_RANGE
            if x_min >= x_max or y_min >= y_max:
                raise ValueError("I limiti SCAN_X_RANGE e SCAN_Y_RANGE devono essere crescenti")
            # COSA: numero di righe: ceil(0.22 / 0.04) + 1 = ceil(5.5) + 1 = 7. ceil arrotonda per ECCESSO,
            #       cosi' la distanza vera fra righe (0.22 / 6 = 36.7 mm) non supera mai SCAN_ROW_SPACING.
            rows = int(np.ceil((y_max - y_min) / p.SCAN_ROW_SPACING)) + 1
            # COSA: le 7 quote y equidistanti: -0.110, -0.073, -0.037, 0, 0.037, 0.073, 0.110.
            y_values = np.linspace(y_min, y_max, rows)
            # COSA: list comprehension a doppio ciclo: per ogni riga (enumerate da' indice row e valore y) due
            #       punti; righe pari da x_min a x_max, righe dispari da x_max a x_min (row % 2 = resto della
            #       divisione per 2). Risultato: 14 waypoint a zig-zag, (-0.16,-0.11) (0.16,-0.11) (0.16,-0.073) ...
            # PERCHÉ: copre tutto il rettangolo senza "salti" in aria: si cambia riga con un segmento corto.
            self.waypoints = np.array([
                (x, y)
                for row, y in enumerate(y_values)
                for x in ((x_min, x_max) if row % 2 == 0 else (x_max, x_min))
            ], dtype=float)
        else:
            raise ValueError(f"SCAN_PATTERN non supportato: {p.SCAN_PATTERN}")
        # COSA: np.diff(..., axis=0) = differenze fra waypoint consecutivi (13 vettori); norm(..., axis=1) = la
        #       lunghezza di ognuno. Misurato: righe lunghe 0.32 m, cambi riga 0.0367 m.
        segment_lengths = np.linalg.norm(np.diff(self.waypoints, axis=0), axis=1)
        # COSA: un segmento lungo 0 darebbe durata 0 e divisione per zero in update().
        if np.any(segment_lengths <= 0):
            raise ValueError("Il percorso di scansione contiene segmenti di lunghezza nulla")
        # COSA: durata di ogni segmento = lunghezza / velocita' media: 21.33 s le righe, 2.44 s i cambi riga.
        self.segment_durations = segment_lengths / p.SCAN_SPEED
        # COSA: somma cumulata = istante di FINE di ogni segmento sul clock della traiettoria: 21.33, 23.78, ..., 164.
        self.segment_end_times = np.cumsum(self.segment_durations)
        # COSA: lunghezza totale (2.46 m) e durata totale T (164 s). DOVE: run_admittance (stima della durata).
        self.length = float(segment_lengths.sum())
        self.T = float(self.segment_end_times[-1])
        # COSA: il "clock" della traiettoria: NON e' il tempo della simulazione, avanza solo con contatto buono.
        self.clock = 0.0
        # COSA: istante del primo contatto (None = non ancora avvenuto).
        self.t_contact = None

    def update(self, t, dt, F_meas):
        p = self.p
        # COSA: la prima volta che la forza misurata supera F_GATE1 (2 N) memorizza l'istante. Misurato: 3.615 s.
        if self.t_contact is None and F_meas > p.F_GATE1:
            self.t_contact = t
        # COSA: il clock avanza di dt SOLO se: c'e' stato il contatto, sono passati SCAN_DELAY (2 s) e la forza
        #       e' sopra F_CONTACT_OK (1 N). min(..., T) lo ferma alla fine. Misurato: parte a 5.615 s.
        # PERCHÉ: se il contatto si perde il riferimento XY si ferma e la sonda non scivola via in aria.
        if self.t_contact is not None and t - self.t_contact >= p.SCAN_DELAY and F_meas > p.F_CONTACT_OK:
            self.clock = min(self.clock + dt, self.T)
        # COSA: percorso finito: ultimo waypoint, velocita' zero, avanzamento 1. .copy() perche' chi riceve
        #       l'array potrebbe modificarlo e non deve toccare self.waypoints.
        if self.clock >= self.T:
            return self.waypoints[-1].copy(), np.zeros(2), 1.0
        # COSA: searchsorted(fini, clock, side="right") = indice del primo istante di fine STRETTAMENTE maggiore del
        #       clock = segmento in corso. Esempio: clock = 22 s -> 21.33 <= 22 < 23.78 -> segmento 1.
        segment = int(np.searchsorted(self.segment_end_times, self.clock, side="right"))
        # COSA: istante d'inizio del segmento = fine del precedente; per il segmento 0 (if segment = "se non e' 0") vale 0.
        segment_start = self.segment_end_times[segment - 1] if segment else 0.0
        duration = self.segment_durations[segment]
        # COSA: avanzamento normalizzato DENTRO il segmento, fra 0 e 1.
        s = (self.clock - segment_start) / duration
        # COSA: vettore del segmento (fine - inizio), in XY.
        delta = self.waypoints[segment + 1] - self.waypoints[segment]
        # COSA: posizione = inizio + delta * min_jerk(s): parte e arriva con velocita' e accelerazione nulle.
        # PERCHÉ: agli angoli della serpentina (90 gradi) la sonda si ferma: niente salti di velocita'.
        xy = self.waypoints[segment] + delta * float(min_jerk(s))
        # COSA: derivata rispetto al tempo (regola della catena): d/dt min_jerk(s) = min_jerk'(s) * ds/dt,
        #       con ds/dt = 1 / duration. Al centro della riga min_jerk'(0.5) = 1.875 -> 0.32 * 1.875 / 21.33 = 28 mm/s.
        # NB: questa velocita' viene restituita anche quando il clock e' fermo (F < 1 N): in quel caso il P del
        #     BLOCCO 4 la compensa. Nella simulazione di default il contatto non si perde mai (0 s misurati).
        ds_dt = float(min_jerk_dot(s)) / duration
        # COSA: restituisce (posizione XY di riferimento, velocita' XY di feedforward, avanzamento 0..1).
        # DOVE: step() BLOCCO 4a: xy -> goal[0], goal[1]; velocita' -> v_ff_xy (BLOCCO 4); avanzamento -> log "scan_s".
        return xy, delta * ds_dt, self.clock / self.T


# =====================================================================================================
#  BLOCCO 5: CINEMATICA INVERSA DIFFERENZIALE
# =====================================================================================================
def differential_ik(J, v6, q, q_null, p: AdmittanceParams):
    """q_dot = J^+ v6 + (I - J^+ J) k_null (q_null - q), con pseudo-inversa smorzata (DLS)."""
    # COSA: numero di giunti = colonne dello Jacobiano (7).
    nv = J.shape[1]
    # COSA: pseudo-inversa smorzata J^+ = J^T (J J^T + lambda^2 I)^-1 (7 x 6), lambda = IK_DAMPING = 0.02.
    # PERCHÉ: trova le velocita' di giunto che realizzano v6 (minimi quadrati); lo smorzamento evita velocita'
    #         enormi vicino alle singolarita'.
    J_pinv = J.T @ np.linalg.inv(J @ J.T + p.IK_DAMPING ** 2 * np.eye(6))
    # COSA: q_dot = J^+ v6 (moto richiesto alla punta) + (I - J^+ J) K_NULL_V (q_null - q) (richiamo verso la
    #       postura iniziale, proiettato nel nullspace = moti dei giunti che non spostano la punta).
    # PERCHÉ: 7 giunti per 6 numeri: il grado in piu' (il gomito) va tenuto vicino a q_null, altrimenti deriva.
    #         Con la DLS il proiettore non e' esatto: una piccolissima parte del richiamo arriva alla punta.
    q_dot = J_pinv @ v6 + (np.eye(nv) - J_pinv @ J) @ (p.K_NULL_V * (q_null - q))
    # COSA: taglia ogni componente a +-1 rad/s (sicurezza; se intervenisse cambierebbe la direzione del moto).
    # DOVE: step() BLOCCO 5: q_dot_cmd -> q_ref += q_dot_cmd * dt, e BLOCCO 6 (velocita' di riferimento del PD).
    return np.clip(q_dot, -p.QDOT_MAX, p.QDOT_MAX)


# =====================================================================================================
#  SIMULAZIONE CON IL CONTROLLO DI AMMETTENZA
# =====================================================================================================
# COSA: la sonda parte 6 cm sopra la pelle a riposo (= D_FAR: si parte con K_d = K_FAR).
# DOVE: __init__ (controllo della posa ps.Q_HOVER, calcolata per questa quota), target().
HOVER = 0.06          # quota di partenza sopra la pelle a riposo [m]
# COSA: il planner punta 8 mm SOTTO la pelle.  DOVE: target(); run_admittance --goal-depth.
# PERCHÉ: se puntasse alla pelle, F_virt = K_d * e_in andrebbe a zero proprio sulla pelle e la sonda si
#         fermerebbe prima di toccare; puntando dentro, la molla spinge finche' K_d non si spegne.
GOAL_DEPTH = 0.008    # profondita' dell'obiettivo del planner sotto la pelle a riposo [m] (serve solo finche' K_d > 0)


class AdmittanceSim:
    """
    Panda in coppia + sonda + tessuto Hunt-Crossley + sensore F/T, con la legge di controllo:
    ammettenza ad integratore puro -> IK differenziale -> PD di posizione di giunto.
    Ogni chiamata a step() e' un passo fisico da 1 ms; i segnali finiscono in self.log.
    """

    # COSA: argomenti con default: parametri (None = default), calibrazione e filtro dell'ATI, durata della discesa
    #       (4 s), profondita' dell'obiettivo, seme del rumore.
    # DOVE: creato da run_admittance.build_sim.
    def __init__(self, params: AdmittanceParams = None, ati_calibration="SI-25-0.25",
                 ati_filter=None, descent_time=4.0, goal_depth=GOAL_DEPTH, seed=0):
        # COSA: target() divide per descent_time: con 0 o negativo si ferma subito con un messaggio chiaro.
        if descent_time <= 0:
            raise ValueError("descent_time deve essere positivo")
        # COSA: "params or AdmittanceParams()": se params e' None usa i default.
        self.p = params or AdmittanceParams()
        # COSA: costruisce la traiettoria (anche senza scansione: controlla i parametri e da' traj.T a run_admittance).
        self.traj = LineTrajectory(self.p)
        # COSA: punto (x0, y0) dove scende la sonda = primo punto del percorso: (-0.16, -0.11) con la serpentina.
        #       Vale anche senza scansione (SCAN_ENABLE = False): la sonda scende li' e ci resta.
        # PERCHÉ: la posa di partenza salvata (ps.Q_HOVER) e' sopra questo punto.
        # DOVE: controllo della posa qui sotto, z_rest, target().
        self.x0, self.y0 = float(self.traj.waypoints[0][0]), float(self.traj.waypoints[0][1])
        self.descent_time, self.goal_depth = float(descent_time), float(goal_depth)

        # scena e tessuto
        # COSA: costruisce il modello MuJoCo (panda_scene.build_model) e gli indici utili.
        self.model, self.ids = ps.build_model()
        # COSA: MjData = lo STATO (qpos, qvel, time) + tutto quello che MuJoCo calcola (posizioni, sensori, ...).
        self.data = mujoco.MjData(self.model)
        # COSA: nomi corti per non scrivere self.model / self.data ogni volta.
        m, d = self.model, self.data
        # COSA: passo di 1 ms (impostato in build_model).  DOVE: ammettenza, traiettoria, q_ref, log.
        self.dt = m.opt.timestep
        # COSA: tessuto coi default (TorsoGeometry, StiffnessMap, Breathing). I sussulti li aggiunge run_admittance
        #       DOPO (sim.tissue.events = gaussian_events(...)).
        # DOVE: step() BLOCCO 0 (contact), BLOCCO 2 (tissue_stiffness), grafica.
        self.tissue = Tissue()

        # posa iniziale: sonda verticale HOVER sopra la pelle nel punto di partenza
        # COSA: carica il keyframe "hover" salvato nel modello (panda_scene.build_model): qpos = ps.Q_HOVER,
        #       qvel = 0, ctrl = 0, time = 0. m.key("hover").id = indice del keyframe col nome "hover" (= 1:
        #       lo 0 e' "home" delle Menagerie). Nessun calcolo: i giunti sono gia' nella posa giusta.
        # PERCHÉ: la simulazione parte gia' sopra la pelle, e da t = 0 lavora solo il controllo di ammettenza.
        mujoco.mj_resetDataKeyframe(m, d, m.key("hover").id)
        # COSA: orientamento desiderato della sonda: le COLONNE sono gli assi della sonda nel mondo:
        #       x_sonda = x_mondo, y_sonda = -y_mondo, z_sonda = -z_mondo (verso il basso). E' una rotazione di
        #       180 gradi attorno a x. La sonda e' lungo +z di link7 (TIP_Z > 0), quindi z giu' = sonda verticale.
        # DOVE: step() BLOCCO 4 (w = KP_ROT_V * rot_err(R, R_des)): la sonda resta verticale
        #       (NON perpendicolare alla pelle) per tutta la simulazione. Q_HOVER e' stata calcolata con questa R_des.
        self.R_des = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1.0]])      # asse z della sonda verso il basso
        # COSA: quota della pelle A RIPOSO (senza respiro) nel punto di partenza: 0.0835 m in (-0.16, -0.11).
        # DOVE: controllo qui sotto e target() (inizio e fine della discesa).
        self.z_rest = float(self.tissue.geom.z0(self.x0, self.y0))
        # COSA: la postura di partenza diventa la postura "preferita": (-0.206, -0.19, -0.072, -2.69, -0.023, 2.5, -0.258).
        # DOVE: step() BLOCCO 5: il nullspace di differential_ik tira i giunti verso q_null per tutta la simulazione.
        self.q_null = d.qpos[:m.nv].copy()                      # postura preferita per il nullspace
        # COSA: ricalcolo di tutte le grandezze (posizioni di corpi e siti, ...) per la posa caricata.
        mujoco.mj_forward(m, d)
        # COSA: distanza fra dove sta davvero la punta e dove dovrebbe stare: 6 cm sopra la pelle in (x0, y0).
        #       Misurato con i default: 0.06 mm (l'errore residuo dell'IK con cui e' stata calcolata Q_HOVER).
        # PERCHÉ: Q_HOVER vale per UN punto. Se cambi l'inizio del percorso o la forma del torso, il robot partirebbe
        #         lontano dal punto di discesa: meglio fermarsi subito con un messaggio chiaro (soglia 1 mm).
        start_err = np.linalg.norm(d.site_xpos[self.ids["site"]] - np.array([self.x0, self.y0, self.z_rest + HOVER]))
        if start_err > 1e-3:
            raise ValueError(f"la posa salvata ps.Q_HOVER non e' {HOVER*100:.0f} cm sopra il punto di partenza "
                             f"({self.x0:.3f}, {self.y0:.3f}): scarto {start_err*1000:.1f} mm. Va ricalcolata.")

        # controllore
        # COSA: crea i pezzi del controllore: soglie (BLOCCO 1), ammettenza (BLOCCO 3, v = 0).
        self.schedule = StiffnessSchedule(self.p)
        self.adm = PureIntegratorAdmittance(self.p.M_D, self.dt, self.p.INTEGRATOR)
        # COSA: il riferimento di giunto parte UGUALE ai giunti veri: nessun salto di coppia al primo passo.
        # DOVE: step() BLOCCHI 5 (integrato) e 6 (PD).
        self.q_ref = d.qpos[:m.nv].copy()                       # riferimento di giunto (integrato da q_dot)
        # COSA: le rigidezze del PD da tupla ad array numpy (per moltiplicare elemento per elemento).
        self.kp_joint = np.array(self.p.KP_JOINT)

        # Sensore di forza: ATI Nano17 + Net F/T alla flangia, a partire dal wrench vero di MuJoCo.
        # COSA: crea il simulatore dell'ATI (ft_sensor.py) con passo 1 ms, calibrazione, filtro e seme del rumore.
        self.ft = ATINanoSensor(self.dt, ati_calibration, ati_filter, seed=seed)
        # Tara ("Bias" del Net F/T) col robot fermo sopra la pelle: ctrl = compensazione di gravita' -> qacc = 0,
        # quindi il sensore vede solo il peso della sonda, che la tara toglie.
        # COSA: d.qfrc_bias = coppie che servono a reggere il robot (gravita'; Coriolis = 0 perche' fermo).
        #       Darle come comando ai motori -> accelerazione nulla (misurato: |qacc| max = 0.0).
        d.ctrl[:] = d.qfrc_bias[:m.nv]
        # COSA: ricalcola tutto con questi ctrl, compresi i sensori force/torque (che dipendono dall'accelerazione).
        mujoco.mj_forward(m, d)
        # COSA: legge i 6 numeri del sensore vero: misurato (0, 0, -2.943, 0, 0, 0) nel frame ft = peso della sonda.
        #       .copy(): sensordata verra' sovrascritto da MuJoCo.
        # DOVE: tara qui sotto, e prima lettura dell'ATI in step() al primo passo.
        self._ft_raw = d.sensordata[self.ids["ft_adr"]:self.ids["ft_adr"] + 6].copy()   # wrench vero (frame ft)
        # COSA: come il pulsante "Bias": offset = media di 100 letture rumorose (misurato Fz offset -2.9436 N).
        #       Da qui in poi, con la sonda appesa e ferma, l'ATI legge ~0.
        self.ft.tare(self._ft_raw)

        # COSA: dizionario costruito con una dict comprehension: per ogni nome una lista vuota.
        # DOVE: step() vi aggiunge un valore per chiave ogni ms; run_admittance lo legge (grafici, CSV, statistiche,
        #       stampe del viewer).
        self.log = {k: [] for k in (
            "t", "fn", "fmeas", "delta", "ncontact", "zs", "ztip", "x", "y", "tau", "ft_saturated",
            "Kd", "Bd", "Kt", "v_in", "v_lim", "dist", "Fdes_eff", "F_virt",
            "x_ref", "y_ref", "z_ref", "scan_s", "exy")}

    def target(self, t):
        """Obiettivo del planner: discesa min-jerk da HOVER sopra la pelle a goal_depth sotto, in descent_time."""
        # COSA: quota di partenza (6 cm sopra la pelle a riposo) e di arrivo (8 mm sotto).
        z0 = self.z_rest + HOVER
        z1 = self.z_rest - self.goal_depth
        # COSA: z = z0 + (z1 - z0) * min_jerk(t / 4 s): discesa liscia in 4 s, poi ferma in z1 (min_jerk taglia
        #       oltre 1). Misurato: z = 0.1435 a t = 0, 0.0755 da t = 4 s in poi.
        # PERCHÉ: e' l'obiettivo della molla virtuale: serve solo finche' K_d > 0 (a contatto F_virt = 0).
        # DOVE: step() BLOCCO 4a: goal (x, y sovrascritti dalla serpentina) -> e_in (BLOCCO 3), e_xy (BLOCCO 4).
        return np.array([self.x0, self.y0, z0 + (z1 - z0) * min_jerk(t / self.descent_time)])

    def step(self):
        # COSA: nomi corti per modello, dati, indici, parametri.
        m, d, ids, p = self.model, self.data, self.ids, self.p
        # COSA: t = istante all'INIZIO del passo (d.time avanzera' di dt dentro mj_step2).
        t, dt = d.time, self.dt
        # COSA: prima meta' del passo di MuJoCo: calcola con qpos, qvel al tempo t le posizioni di corpi e siti,
        #       la matrice d'inerzia, qfrc_bias (gravita' + Coriolis) e i sensori di posizione/velocita'.
        #       NON integra e NON calcola i sensori di forza.
        # PERCHÉ: mj_step = mj_step1 + mj_step2. Spezzarlo permette di mettere IN MEZZO il controllore: leggo le
        #         posizioni al tempo t, calcolo forza del tessuto e coppie, poi mj_step2 le usa.
        mujoco.mj_step1(m, d)                                   # posizioni, Jacobiani, M, qfrc_bias al tempo t

        # ---------------------------------------------------------------- BLOCCO 0: misura
        # COSA: posizione della punta nel mondo. .copy(): dopo mj_step2 l'array di MuJoCo cambia; serve il valore a t.
        # DOVE: Tissue.contact, apply_tip_force, K_t (pos[0], pos[1]), e_in, e_xy, log x, y, ztip.
        pos = d.site_xpos[ids["site"]].copy()
        # COSA: orientamento della punta (3 x 3). Senza copy e' una VISTA sulla memoria di MuJoCo: va bene perche'
        #       viene usata prima di mj_step2.  DOVE: BLOCCO 4 (rot_err).
        R = d.site_xmat[ids["site"]].reshape(3, 3)
        # COSA: Jacobiano 6 x 7 della punta al tempo t.  DOVE: v_tip qui sotto e differential_ik (BLOCCO 5).
        J = ps.jac(m, d, ids["site"])
        # COSA: velocita' dei 7 giunti (vista).  DOVE: v_tip e il termine D del PD (BLOCCO 6).
        qd = d.qvel[:m.nv]
        # COSA: velocita' 6D della punta = J @ q_dot: v_tip[:3] lineare [m/s], v_tip[3:] angolare [rad/s].
        v_tip = J @ qd
        # COSA: forza del tessuto con Hunt-Crossley (tissue.py): usa la posizione e la velocita' VERE della punta.
        #       Restituisce force (vettore), fn (modulo), delta, ddelta, k_eq, normal, zs.
        # PERCHÉ: e' la "fisica del paziente": il controllore non la vede, la vede solo attraverso il sensore.
        hc = self.tissue.contact(pos, v_tip[:3], t)             # forza del tessuto (Hunt-Crossley) sulla punta
        # COSA: scrive la forza in d.xfrc_applied del corpo "probe", applicata nel punto pos (con la sua coppia).
        # DOVE: mj_step2 qui sotto: muove il robot E finisce nel sensore force/torque.
        ps.apply_tip_force(d, ids["probe_body"], hc.force, pos)  # xfrc_applied: la vede anche il sensore F/T
        # COSA: normale uscente (lunghezza 1). -n = direzione "dentro la pelle".
        # DOVE: F_meas, e_in, v_tan, v6.
        n = hc.normal                                           # normale uscente della pelle sotto la punta
        # COSA: distanza dalla pelle lungo la normale = -delta: > 0 sopra, < 0 dentro. A regime ~ -8 mm al punto di
        #       partenza (dentro la pelle).
        # NB: il controllore usa qui la conoscenza del modello della pelle; su un robot vero d andrebbe stimata
        #     (per questo esiste anche il cancello in forza F_GATE).
        # DOVE: BLOCCO 1 (K_d, rho), v_in_max (BLOCCO 3), log "dist".
        dist = -hc.delta                                        # distanza dalla pelle lungo n (>0 = sopra)

        # UNICO segnale di forza visto dal controllore: wrench ATI tarato nel frame del sensore;
        # _ft_raw e' quello calcolato da MuJoCo nel passo precedente
        # COSA: catena dell'ATI (ft_sensor.read): rumore -> filtro -> saturazione -> ritardo -> meno la tara.
        #       In ingresso il wrench calcolato da mj_step2 del passo PRECEDENTE: 1 ms di ritardo intrinseco, come
        #       sul robot vero, dove la lettura arriva dopo il fatto.
        w_meas = self.ft.read(self._ft_raw)
        # COSA: orientamento del frame del sensore: R_ft @ v porta un vettore dal frame del sensore al mondo.
        R_ft = d.site_xmat[ids["ft_site"]].reshape(3, 3)
        # il sensore misura la forza che la flangia esercita sulla sonda: dopo la tara vale -F_tessuto
        # COSA: w_meas[:3] = forza nel frame del sensore; R_ft @ ... = nel mondo; @ n = componente lungo la normale;
        #       il meno cambia verso: la flangia spinge la sonda CONTRO il tessuto (-F_tessuto), quindi
        #       -(...) . n = forza del tessuto lungo n, positiva quando si preme. float() -> numero Python.
        #       Misurato a t = 6 s: F_meas 4.991 N contro forza vera fn 4.994 N.
        # DOVE: BLOCCHI 1 (gate), 3 (e_F), 4a (clock della traiettoria), log "fmeas".
        F_meas = float(-(R_ft @ w_meas[:3]) @ n)

        # ---------------------------------------------------------------- BLOCCO 1: soglie di K_d
        # COSA: rigidezza della molla virtuale per questa distanza e forza (StiffnessSchedule.K).
        K_d = self.schedule.K(dist, F_meas)
        # COSA: frazione di F_DES attiva (0 lontano, 1 a contatto).
        rho = self.schedule.rho(dist, F_meas)
        # COSA: forza desiderata effettiva.  DOVE: e_F (BLOCCO 3), log "Fdes_eff".
        F_des_eff = rho * p.F_DES

        # ---------------------------------------------------------------- BLOCCO 2: smorzamento critico
        # COSA: rigidezza del tessuto sotto la punta (pos[0] = x, pos[1] = y). Calcolata anche in aria: B_d vale gia'
        #       ~13.7 Ns/m durante la discesa (misurato).
        K_t = tissue_stiffness(self.tissue, pos[0], pos[1], p.F_DES, p.KT_MODE)
        # COSA: B_d = max(2 sqrt(M_D (K_t + eventuale K_d)), B_MIN). "(K_d if cond else 0.0)" = espressione condizionale.
        # DOVE: adm.step (BLOCCO 3), log "Bd".
        B_d = critical_damping(p.M_D, K_t + (K_d if p.B_INCLUDE_KD else 0.0), p.B_MIN)

        # ---------------------------------------------------------------- BLOCCO 4a: traiettoria XY
        # COSA: obiettivo del planner (array nuovo a ogni chiamata, quindi si puo' modificare).
        goal = self.target(t)                                   # obiettivo del planner (serve solo finche' K_d > 0)
        if p.SCAN_ENABLE:
            # COSA: riferimento XY, velocita' XY di feedforward e avanzamento dalla serpentina (LineTrajectory.update).
            xy_ref, v_ff_xy, scan_s = self.traj.update(t, dt, F_meas)
            # COSA: sostituisce x e y dell'obiettivo con quelli della serpentina (spacchetta l'array di 2 numeri).
            goal[0], goal[1] = xy_ref                           # la quota z del planner resta quella della discesa
        else:
            # COSA: senza scansione: nessuna velocita' di feedforward, avanzamento 0; goal resta (x0, y0).
            v_ff_xy, scan_s = np.zeros(2), 0.0

        # ---------------------------------------------------------------- BLOCCO 3: ammettenza + Eulero
        # COSA: (goal - pos) . (-n) = quanto l'obiettivo e' piu' "dentro" della punta, lungo la direzione verso la pelle.
        #       > 0 = l'obiettivo e' piu' in basso, la molla tira verso la pelle.
        e_in = float((goal - pos) @ (-n))                       # distanza dall'obiettivo lungo l'asse verso la pelle
        # COSA: forza della molla virtuale [N]. Misurato: 0.078 N a t = 0.5 s, 0.66 a 2.0 s, 0 a contatto.
        F_virt = K_d * e_in                                     # forza virtuale della molla (0 a contatto)
        # COSA: errore di forza: > 0 = premo troppo poco -> la punta deve avanzare.
        e_F = F_des_eff - F_meas                                # errore di forza
        # COSA: integra M_d v' + B_d v = F_virt + e_F per un passo -> nuova velocita' verso la pelle.
        v_in = self.adm.step(F_virt + e_F, B_d)
        # COSA: limite superiore di velocita' per questa distanza/forza.
        v_lim = self.schedule.v_in_max(dist, F_meas)
        # COSA: limita v_in in [-V_RETRACT, v_lim] e lo riscrive nello stato (anti-windup).
        #       Misurato: 20 mm/s (= V_TOUCH) in avvicinamento a 2.9 s, 5.7 mm/s a 4.0 s, pochi mm/s a regime.
        # DOVE: BLOCCO 4 (v6 = -n * v_in + ...), log "v_in".
        v_in = self.adm.clamp(-p.V_RETRACT, v_lim)

        # ---------------------------------------------------------------- BLOCCO 4: velocita' cartesiana 6D
        # velocita' TANGENTE alla pelle: feedforward della linea + correzione P sull'errore in XY.
        # La componente z si ricava imponendo v_tan . n = 0, cosi' la sonda segue la curvatura del busto.
        # COSA: errore di posizione in XY fra obiettivo e punta.  DOVE: v_xy, log "exy".
        e_xy = np.array([goal[0] - pos[0], goal[1] - pos[1]])
        # COSA: velocita' XY = feedforward della serpentina + KP_POS_V * errore (anello P che corregge la deriva).
        v_xy = v_ff_xy + p.KP_POS_V * e_xy
        # COSA: vz scelta perche' la velocita' sia TANGENTE alla pelle: v . n = 0 ->
        #       n0 vx + n1 vy + n2 vz = 0 -> vz = -(n0 vx + n1 vy) / n2  (n2 = coseno dell'inclinazione, mai 0 qui).
        # PERCHÉ: lo spostamento in XY non deve cambiare la compressione: quella la decide solo v_in lungo -n.
        v_tan = np.array([v_xy[0], v_xy[1], -(n[0] * v_xy[0] + n[1] * v_xy[1]) / n[2]])
        # COSA: se |v_tan| > V_TAN_MAX la scala a V_TAN_MAX mantenendo la direzione; altrimenti fattore 1.
        #       max(..., 1e-9) evita la divisione per zero quando v_tan = 0.
        v_tan = v_tan * min(1.0, p.V_TAN_MAX / max(np.linalg.norm(v_tan), 1e-9))
        # COSA: velocita' angolare = KP_ROT_V * errore di orientamento (ps.rot_err), tagliata a +-W_MAX per componente.
        # PERCHÉ: tiene la sonda verticale (R_des) anche quando la forza del tessuto la fa ruotare.
        w = np.clip(p.KP_ROT_V * ps.rot_err(R, self.R_des), -p.W_MAX, p.W_MAX)
        # COSA: comando 6D della punta: [velocita' lineare ; angolare]. -n * v_in = moto lungo la normale (verso la
        #       pelle se v_in > 0) + v_tan = moto sulla superficie.
        # PERCHÉ: separazione degli assi: forza lungo n (ammettenza), posizione nel piano tangente (serpentina),
        #         orientamento con un P.
        # DOVE: BLOCCO 5 (differential_ik).
        v6 = np.concatenate([-n * v_in + v_tan, w])

        # ---------------------------------------------------------------- BLOCCO 5: IK differenziale
        # COSA: posizione dei giunti al tempo t (vista su d.qpos, usata prima di mj_step2).
        q = d.qpos[:m.nv]
        # COSA: velocita' di giunto che realizzano v6, piu' il richiamo alla postura q_null.
        q_dot_cmd = differential_ik(J, v6, q, self.q_null, p)
        # COSA: integrazione in giunto: q_ref(t + dt) = q_ref(t) + q_dot * dt. E' qui che la velocita' diventa
        #       una POSIZIONE di riferimento (la seconda integrazione; l'ammettenza integra una volta sola).
        self.q_ref = self.q_ref + q_dot_cmd * dt                # integrazione in giunto -> riferimento di posizione
        # COSA: anti-windup: q_ref - q tagliato a +-0.08 rad, poi q_ref = q + differenza tagliata.
        # PERCHÉ: se il robot non riesce a seguire (bloccato), q_ref non scappa via; e poiche' la coppia del PD e'
        #         KP * (q_ref - q), limita anche la coppia (600 * 0.08 = 48 Nm). Misurato in contatto: 0.0036 rad.
        self.q_ref = q + np.clip(self.q_ref - q, -p.Q_REF_MAX_ERR, p.Q_REF_MAX_ERR)   # anti-windup

        # ---------------------------------------------------------------- BLOCCO 6: PD di posizione di giunto
        # COSA: matrice 7 x 7 vuota; mj_fullM ci copia la matrice d'inerzia dei giunti (MuJoCo la tiene in forma
        #       compatta, gia' calcolata in mj_step1).
        M = np.zeros((m.nv, m.nv))
        mujoco.mj_fullM(m, d, M)                                # matrice d'inerzia dei giunti
        # COSA: Kd_i = 2 zeta sqrt(Kp_i M_ii): la STESSA formula di critical_damping (BLOCCO 2), giunto per giunto,
        #       con la molla KP_JOINT e l'inerzia vista dal giunto i (diagonale di M).
        #       Misurato alla partenza: M_ii = 0.79 1.31 0.81 1.01 0.125 0.145 0.105 (comprende l'armatura 0.1)
        #       -> Kd = 43.6 56.2 44.1 49.2 11.2 9.3 4.6 Nms/rad.
        kd_joint = 2.0 * p.ZETA_JOINT * np.sqrt(self.kp_joint * np.diag(M))
        # COSA: PD: molla verso la posizione di riferimento + smorzatore verso la VELOCITA' di riferimento
        #       (q_dot_cmd, non 0: cosi' non frena il moto comandato).
        tau = self.kp_joint * (self.q_ref - q) + kd_joint * (q_dot_cmd - qd)
        # COSA: + qfrc_bias = compensazione di gravita' e Coriolis (stessa idea della tara): il PD deve occuparsi
        #       solo degli scostamenti. Taglio ai limiti del Franka e scrittura nei comandi dei motori (gear 1 ->
        #       ctrl = coppia in Nm). Misurato: coppia massima 19-23 Nm.
        d.ctrl[:] = np.clip(tau + d.qfrc_bias[:m.nv], -ps.TORQUE_LIMITS, ps.TORQUE_LIMITS)
        # COSA: seconda meta' del passo: forze degli attuatori (ctrl) + forze esterne (xfrc_applied) -> accelerazioni,
        #       sensori force/torque, poi integrazione: qpos, qvel, time diventano quelli di t + dt.
        mujoco.mj_step2(m, d)                                   # dinamica + sensori -> stato al tempo t + dt
        # COSA: salva il wrench vero appena calcolato (6 numeri; .copy() perche' sensordata sara' sovrascritto).
        # DOVE: letto dall'ATI all'INIZIO del passo successivo (self.ft.read): ecco il ritardo di 1 passo.
        self._ft_raw = d.sensordata[ids["ft_adr"]:ids["ft_adr"] + 6].copy()   # wrench vero per la prossima lettura

        # ---------------------------------------------------------------- log
        # COSA: tupla di coppie (nome, valore); per ognuna aggiunge float(valore) alla lista self.log[nome].
        #       Particolari: "delta" = max(delta, 0) (penetrazione, 0 fuori); "ncontact" = forza vera > 1 mN
        #       (True/False -> 1.0/0.0); "tau" = coppia piu' grande in modulo; "exy" = hypot = |e_xy|.
        # DOVE: run_admittance (analyse_and_plot, run_viewer).
        for k, val in (("t", t), ("fn", hc.fn), ("fmeas", F_meas), ("delta", max(hc.delta, 0.0)),
                       ("ncontact", hc.fn > 1e-3), ("zs", hc.zs), ("ztip", pos[2]), ("x", pos[0]), ("y", pos[1]),
                       ("tau", np.abs(d.ctrl).max()), ("ft_saturated", self.ft.saturated),
                       ("Kd", K_d), ("Bd", B_d), ("Kt", K_t), ("v_in", v_in), ("v_lim", v_lim), ("dist", dist),
                       ("Fdes_eff", F_des_eff), ("F_virt", F_virt), ("x_ref", goal[0]), ("y_ref", goal[1]),
                       ("z_ref", goal[2]), ("scan_s", scan_s), ("exy", np.hypot(*e_xy))):
            self.log[k].append(float(val))
        # COSA: restituisce la forza VERA del tessuto.  DOVE: run_viewer (colore della punta e stampa "F vera").
        return hc.fn
