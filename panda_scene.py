"""
Scena MuJoCo: Franka Panda (controllo in coppia) + sonda con sensore F/T + superficie del torso (solo grafica).

Il contatto NON e' calcolato da MuJoCo: la forza del tessuto viene da Tissue.contact() (Hunt-Crossley) e
viene applicata alla sonda con apply_tip_force(). Il torso e' un heightfield senza collisioni, aggiornato
solo per la grafica.

Prerequisito: modello Panda dalle Menagerie
    git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git
e imposta PANDA_DIR qui sotto (o la variabile d'ambiente PANDA_DIR).
"""
# =====================================================================================================
#  COME SONO SCRITTI I COMMENTI DI STUDIO
#    COSA   = cosa fa la riga, pezzo per pezzo (sintassi Python/numpy, funzione MuJoCo, matematica)
#    PERCHÉ = a cosa serve qui e cosa succederebbe senza
#    DOVE   = chi usa il risultato nella simulazione (file -> funzione -> blocco)
#  I numeri tra parentesi sono MISURATI eseguendo il codice (simulazione di default, 13-11-2026).
#
#  MAPPA DEL FILE (chi chiama cosa)
#    build_model()                 <- admittance_control.AdmittanceSim.__init__: crea robot, sonda, sensore, torso
#                                     e il keyframe "hover" (posa di partenza Q_HOVER)
#    rot_err(), jac()              <- AdmittanceSim.step() (BLOCCHI 0, 4, 5)
#    apply_tip_force()             <- AdmittanceSim.step(), BLOCCO 0: mette la forza del tessuto dentro la fisica
#    update_tissue_mesh(),
#    measure_tissue_compression()  <- run_admittance.run_viewer(): SOLO grafica, non toccano la fisica
# =====================================================================================================

# COSA: modulo della libreria standard per leggere le variabili d'ambiente del sistema (os.environ).
# PERCHÉ: il percorso del modello Panda si puo' cambiare da fuori senza toccare il codice.  DOVE: PANDA_DIR.
import os
# COSA: Path = oggetto "percorso"; l'operatore / unisce cartella e nome file (PANDA_DIR / "panda_nohand.xml").
# PERCHÉ: piu' sicuro che concatenare stringhe a mano.  DOVE: PANDA_DIR e build_model().
from pathlib import Path
# COSA: numpy = array e algebra lineare (np.cross, np.linalg.inv, ...).  DOVE: ovunque in questo file.
import numpy as np
# COSA: binding Python di MuJoCo: MjSpec (modello modificabile), MjModel (modello compilato), MjData (stato).
import mujoco

# COSA: os.environ.get(chiave, default) restituisce la variabile d'ambiente PANDA_DIR se esiste, altrimenti la
#       stringa fissa; Path(...) la trasforma in percorso.
# PERCHÉ: il Panda NON e' nel repo, sta nelle Menagerie di DeepMind (cartella mujoco_menagerie).
# DOVE: build_model(), prima riga: MjSpec.from_file(PANDA_DIR / "panda_nohand.xml").
PANDA_DIR = Path(os.environ.get("PANDA_DIR", "/home/campri/Scrivania/Robotics/mujoco_menagerie/franka_emika_panda"))

# griglia grafica del torso (heightfield)
# COSA: numero di punti della griglia del torso: 81 colonne lungo x, 61 righe lungo y. Con le semi-dimensioni
#       SX, SY qui sotto il passo e' 2*0.20/80 = 5 mm in x e 2*0.15/60 = 5 mm in y (81*61 = 4941 quote).
# PERCHÉ: 5 mm bastano a vedere respiro e avvallamento sotto la sonda, e 4941 numeri si aggiornano in fretta
#         a ogni fotogramma.
# DOVE: build_model() (nrow, ncol dell'heightfield), _gx/_gy qui sotto, update_tissue_mesh() (fetta NR*NC).
NC, NR = 81, 61                 # colonne (x), righe (y)
# COSA: semi-dimensioni della patch: x da -0.20 a +0.20 m (40 cm, asse cranio-caudale), y da -0.15 a +0.15 m.
# PERCHÉ: deve coprire tutta la serpentina (x +-0.16, y +-0.11, AdmittanceParams) con un po' di margine.
# DOVE: size dell'heightfield in build_model(), griglie _gx/_gy.
SX, SY = 0.20, 0.15             # semi-dimensioni patch [m]
# COSA: MuJoCo descrive un heightfield con size = [SX, SY, ELEV, BASE]: la quota di ogni punto e' dato*ELEV con
#       dato fra 0 e 1, quindi ELEV = 0.25 m e' la quota massima disegnabile; BASE = 0.05 m e' lo spessore del
#       "piedistallo" che MuJoCo disegna sotto lo zero.
# PERCHÉ: il torso arriva a H = 0.10 m (+ respiro + sussulti): con 0.25 c'e' tutto, con molto margine.
# DOVE: build_model() (size e piano del pavimento a -BASE), update_tissue_mesh() (z / ELEV).
ELEV, BASE = 0.25, 0.05         # quota massima rappresentabile e spessore sotto lo zero [m]

# COSA: posizione nel mondo della base del robot (corpo link0): 55 cm "dietro" il centro del torso, lungo -x.
# PERCHÉ: il torso e' centrato in (0, 0); la base va messa a una distanza da cui il braccio, con la sonda
#         verticale, raggiunge tutta la patch (commento a destra). Misurato: alla posa home la punta e' a
#         x = 0.0045 m, cioe' a 0.554 m dall'asse del giunto 1 (e' la colonna 0 dello Jacobiano, vedi jac()).
# DOVE: build_model(), spec.body("link0").pos.
PANDA_BASE_POS = (-0.55, 0.0, 0.0)   # con -0.40 il Panda (sonda verticale) non raggiunge x < -80 mm; con -0.55 tutta la patch
# COSA: distanza dall'origine di link7 alla flangia del Panda, lungo l'asse z di link7: 0.107 m (nel file XML
#       delle Menagerie c'e' il corpo "attachment" proprio a pos="0 0 0.107").
# PERCHÉ: sulla flangia del robot vero si monta il sensore ATI, e dopo il sensore la sonda.
# DOVE: sito "ft" (dove misura il sensore), inizio dei cilindri della sonda, TIP_Z.
FLANGE_Z = 0.107                # flangia del Panda lungo l'asse z di link7: qui e' montato il sensore F/T [m]
# COSA: la punta della sonda sta 15 cm oltre la flangia: TIP_Z = 0.257 m lungo l'asse z di link7.
# PERCHÉ: e' IL punto controllato: tutta l'ammettenza lavora sulla posizione e velocita' di questo punto.
# DOVE: sito "tip" in build_model() -> ids["site"] -> jac(), step() BLOCCO 0 (pos -> Tissue.contact).
TIP_Z = FLANGE_Z + 0.15         # punta della sonda lungo l'asse z di link7 [m]
# COSA: massa della sonda: peso = 0.30 * 9.81 = 2.943 N.
# PERCHÉ: il sensore vero vede anche il peso della sonda; la tara lo toglie (misurato: col robot fermo il sensore
#         legge Fz = -2.943 N nel suo frame), e quando la sonda accelera il sensore vede anche la sua inerzia.
# DOVE: cilindro invisibile con massa in build_model().
PROBE_MASS = 0.30               # massa della sonda (porta-sonda incluso) [kg]
# COSA: raggio della sfera verde in fondo alla sonda: 22 mm.
# PERCHÉ: e' SOLO grafica: il contatto fisico (Tissue.contact) usa il PUNTO "tip", che e' il fondo della sfera.
# DOVE: geom "tip_vis", probe_end, update_tissue_mesh() (larghezza dell'avvallamento), measure_tissue_compression().
TIP_RADIUS = 0.022              # raggio della sfera terminale [m]
# COSA: coppie massime dei 7 giunti del Franka [Nm]: 87 per i giunti 1-4 (spalla/gomito), 12 per il polso 5-7.
# PERCHÉ: limiti del robot vero; misurato: in contatto la coppia massima chiesta e' circa 19-23 Nm (giunto 2/4).
# DOVE: ctrlrange degli attuatori qui sotto e step() BLOCCO 6: np.clip(tau + qfrc_bias, -LIMITI, +LIMITI).
TORQUE_LIMITS = np.array([87, 87, 87, 87, 12, 12, 12], dtype=float)
# COSA: angoli dei 7 giunti [rad] della posa di partenza: punta della sonda in (-0.16, -0.11, 0.1435), cioe'
#       6 cm (HOVER) sopra la pelle a riposo nel primo punto della serpentina, sonda verticale (asse z giu').
#       Calcolati UNA volta, offline, con l'IK iterativa (pseudo-inversa smorzata + postura home nel nullspace,
#       400 iterazioni) che stava in questo file fino al commit 679dde9: errore residuo 6.0e-05 m.
#       Scritti con tutte le cifre: la simulazione e' identica, al bit, a quella che faceva l'IK a ogni avvio.
# PERCHÉ: l'IK non era una fase della simulazione (girava prima di t = 0 e "teletrasportava" i giunti): tanto
#         vale salvare il risultato e partire da qui. Il prezzo: vale per UN solo punto di partenza. Se cambi
#         l'inizio del percorso (--scan-x, --scan-y, --scan-start) o la forma del torso, va ricalcolata
#         (AdmittanceSim.__init__ se ne accorge e si ferma con un errore).
# DOVE: build_model() -> keyframe "hover" -> AdmittanceSim.__init__ (mj_resetDataKeyframe), che la usa anche come
#       postura preferita q_null del nullspace (step() BLOCCO 5).
Q_HOVER = (-0.2060103872267039, -0.19019021558822838, -0.07189872951261951, -2.6899356513525126,
           -0.022682449347206736, 2.5000783113337017, -0.25844787553135584)


def build_model():
    # COSA: MjSpec.from_file legge il file XML del Panda e lo trasforma in un MjSpec, cioe' una descrizione del
    #       modello che si puo' ancora modificare da Python (aggiungere/togliere corpi, attuatori, sensori).
    #       str(...) perche' from_file vuole una stringa, non un Path.
    # PERCHÉ: dobbiamo modificare il Panda (attuatori in coppia, sonda, sensore, torso) PRIMA di compilarlo.
    # DOVE: spec viene modificato in tutta la funzione e compilato in fondo (spec.compile()).
    spec = mujoco.MjSpec.from_file(str(PANDA_DIR / "panda_nohand.xml")) #trasformo il file xml in un oggetto MjSpec (modello MuJoCo) modificabile in python


    # sostituisco sette servo di posizione, che convertivano un angolo in coppia tramite un PD, con sette motori che prendono la coppia direttamente.
    # COSA: nel file XML ci sono 7 servo di posizione "actuator1..7" (misurato, giunto 1: coppia =
    #       4500*(ctrl - q) - 450*q_dot). list(...) fa una COPIA della lista: cancellare elementi mentre si scorre
    #       la lista originale ne farebbe saltare qualcuno. spec.delete(a) toglie l'attuatore dal modello.
    # PERCHÉ: il PD di giunto lo facciamo NOI nel BLOCCO 6 di step() (con la compensazione di gravita'): due PD
    #         in serie (il nostro + quello del servo) non avrebbero senso.
    for a in list(spec.actuators):
        spec.delete(a)
    # COSA: ciclo sui 7 giunti (i = 0..6); spec.add_actuator() crea un attuatore nuovo coi valori di default di
    #       MuJoCo: guadagno 1 e nessun termine di bias, cioe' un "motor": forza dell'attuatore = ctrl.
    for i in range(7):
        a = spec.add_actuator()
        # COSA: nome "tau1".."tau7" (f-string: {i + 1} viene sostituito dal numero).
        # PERCHÉ: solo leggibilita' (si vede nel viewer); nel codice gli attuatori si usano per indice, d.ctrl[0..6].
        a.name = f"tau{i + 1}"
        # COSA: il giunto su cui agisce l'attuatore i: "joint1".."joint7".
        a.target = f"joint{i + 1}"
        # COSA: tipo di trasmissione JOINT: la forza dell'attuatore va direttamente sul grado di liberta' del giunto
        #       (giunto rotoidale -> e' una coppia in Nm).
        a.trntype = mujoco.mjtTrn.mjTRN_JOINT  #applica la coppia direttamente al giunto
        # COSA: rapporto di trasmissione 1: coppia sul giunto = gear * forza attuatore = 1 * ctrl.
        # PERCHÉ: cosi' d.ctrl[i] e' direttamente la coppia in Nm.  DOVE: step() BLOCCO 6 scrive d.ctrl[:] = coppie.
        a.gear[0] = 1.0
        # COSA: ctrllimited = 1 accende il limite su ctrl; ctrlrange = [-limite, +limite] del giunto i:
        #       MuJoCo taglia da solo ogni ctrl fuori da questo intervallo.
        # PERCHÉ: seconda sicurezza (step() BLOCCO 6 fa gia' np.clip con gli stessi TORQUE_LIMITS).
        a.ctrllimited = 1
        a.ctrlrange = [-TORQUE_LIMITS[i], TORQUE_LIMITS[i]]

    # COSA: spec.body("link0") cerca nel modello il corpo che si chiama "link0" (la base del robot); .pos e' la
    #       sua posizione rispetto al genitore, che qui e' il mondo. Gli assegno PANDA_BASE_POS = (-0.55, 0, 0).
    # PERCHÉ: sposta tutto il robot (i corpi figli seguono) per avere il torso raggiungibile (vedi PANDA_BASE_POS).
    spec.body("link0").pos = PANDA_BASE_POS  # con spec.body rendo modificabile la posizione del Panda (per centrare la patch del torso sotto la sonda)


    # COSA: prendo il corpo "link7", l'ultimo del braccio (ruota col giunto 7; la flangia e' a FLANGE_Z sul suo asse z).
    link7 = spec.body("link7")
    # COSA: add_body crea un corpo FIGLIO di link7, senza giunti -> saldato a link7 (si muove con lui). Senza pos
    #       e orientamento ha lo stesso frame di link7: le quote FLANGE_Z, TIP_Z sono lungo l'asse z di link7.
    # PERCHÉ un corpo SEPARATO e non geometrie attaccate a link7:
    #   1) i sensori force/torque di MuJoCo misurano la forza che un corpo scambia col suo GENITORE: serve
    #      un'interfaccia link7 <-> probe, che e' proprio il sensore ATI fra flangia e sonda;
    #   2) link7 nelle Menagerie ha un <inertial> esplicito (massa 0.7355 kg): masse di geometrie aggiunte
    #      direttamente a link7 verrebbero ignorate. Nel corpo "probe" la massa conta (misurato: 0.30 kg).
    # DOVE: ids["probe_body"] (= 10) -> apply_tip_force() scrive la forza del tessuto su QUESTO corpo.
    probe = link7.add_body(name="probe")                     # stesso frame di link7, quindi saldato a link7 e si muove con esso.
    # Massa della sonda (0.30 kg). Invisibile: arriva fino al fondo della sfera e il suo bordo
    # piatto sporgerebbe sotto di essa.
    # COSA: cilindro di raggio 0.018 m e semi-altezza 0.075 m (alto 15 cm), centrato a FLANGE_Z + 0.075 = 0.182:
    #       va dalla flangia (0.107) alla punta (0.257). Ha tutta la massa PROBE_MASS, uniforme -> baricentro
    #       della sonda a 0.182, cioe' 7.5 cm sopra la punta (misurato: xipos - punta = 0.075 m).
    #       contype=0, conaffinity=0: nessuna collisione. rgba con alfa 0: invisibile.
    # PERCHÉ: da' alla sonda massa e inerzia realistiche (le vede il sensore F/T). Niente collisioni perche' la
    #         forza di contatto la calcola Tissue.contact(), non MuJoCo.
    # DOVE: il baricentro (d.xipos[probe]) serve ad apply_tip_force() per il braccio della coppia.
    probe.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.018, 0.075, 0],
                   pos=[0, 0, FLANGE_Z + 0.075], mass=PROBE_MASS, contype=0, conaffinity=0,
                   rgba=[0.15, 0.15, 0.18, 0])
    # Corpo visibile della sonda (senza massa): finisce al centro della sfera, cosi' l'emisfero
    # inferiore e' il punto piu' basso e tocca il tessuto.
    # COSA: quota del centro della sfera: 0.257 - 0.022 = 0.235 m.
    probe_end = TIP_Z - TIP_RADIUS  #centro della sfera terminale della sonda (sfera di raggio 0.022 m)
    # COSA: cilindro VISIBILE dalla flangia al centro della sfera: semi-altezza (0.235 - 0.107)/2 = 0.064 m,
    #       centro a (0.107 + 0.235)/2 = 0.171 m. mass=0.0: solo grafica (la massa e' gia' nel cilindro invisibile).
    probe.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.018, (probe_end - FLANGE_Z) / 2, 0],
                   pos=[0, 0, (FLANGE_Z + probe_end) / 2], mass=0.0, contype=0, conaffinity=0,
                   rgba=[0.15, 0.15, 0.18, 1])
    # COSA: sfera verde di raggio TIP_RADIUS centrata in probe_end: il suo punto piu' basso e' TIP_Z.
    #       Senza massa e senza collisioni.
    # PERCHÉ: grafica della punta; il suo colore cambia con la forza (run_viewer).
    # DOVE: ids["tip_geom"] (= 63) -> run_viewer (colore), measure_tissue_compression() (centro e raggio).
    probe.add_geom(name="tip_vis", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[TIP_RADIUS, 0, 0],
                   pos=[0, 0, TIP_Z - TIP_RADIUS], mass=0.0, contype=0, conaffinity=0,
                   rgba=[0.2, 0.8, 0.3, 1])
    # COSA: un SITO e' un punto con un frame, senza massa ne' fisica: qui il punto "tip" a TIP_Z (fondo della sfera),
    #       disegnato come pallino giallo di 3 mm.
    # PERCHÉ: e' il punto di cui MuJoCo calcola posizione (site_xpos), orientamento (site_xmat) e Jacobiano.
    # DOVE: ids["site"] (= 1) -> AdmittanceSim.__init__ (controllo della posa di partenza); step() BLOCCO 0:
    #       pos, R, J -> Tissue.contact, apply_tip_force, IK differenziale.
    probe.add_site(name="tip", pos=[0, 0, TIP_Z], size=[0.003, 0, 0], rgba=[1, 1, 0, 1])
    # Sensore F/T sulla flangia: wrench che link7 scambia con la sonda, nel frame del sito "ft".
    # COSA: sito "ft" sulla flangia (quota FLANGE_Z), pallino magenta. Ha lo stesso orientamento di link7.
    # PERCHÉ: e' il punto e il frame in cui MuJoCo esprime forza e coppia del sensore (come il frame dell'ATI vero).
    # DOVE: i due sensori qui sotto; ids["ft_site"] (= 2) -> step() BLOCCO 0: R_ft porta la forza nel mondo.
    probe.add_site(name="ft", pos=[0, 0, FLANGE_Z], size=[0.004, 0, 0], rgba=[1, 0, 1, 1])
    # COSA: sensore di FORZA (3 numeri) sul sito "ft": la forza che il genitore (link7, la flangia) esercita sul
    #       corpo del sito (probe), espressa nel frame del sito. Misurato, robot fermo: (0, 0, -2.943) nel frame
    #       ft = (0, 0, +2.943) nel mondo: la flangia regge il peso della sonda verso l'alto.
    # PERCHÉ: e' il "wrench vero" da cui ft_sensor.ATINanoSensor simula la lettura dell'ATI.
    # DOVE: d.sensordata[0:3] -> AdmittanceSim._ft_raw (dopo mj_step2) -> self.ft.read() nel passo dopo.
    spec.add_sensor(name="ft_force", type=mujoco.mjtSensor.mjSENS_FORCE,
                    objtype=mujoco.mjtObj.mjOBJ_SITE, objname="ft")
    # COSA: sensore di COPPIA (3 numeri) sullo stesso sito: momento scambiato fra flangia e sonda attorno al sito.
    #       Misurato, sonda appesa: 0 (il baricentro e' sull'asse). In contatto la forza del tessuto agisce sulla
    #       punta, 15 cm sotto la flangia: se la pelle e' inclinata la forza non e' lungo l'asse e la coppia non e' 0.
    # DOVE: d.sensordata[3:6] -> ATI (componenti Tx, Ty, Tz; il controllore usa solo la forza, w_meas[:3]).
    spec.add_sensor(name="ft_torque", type=mujoco.mjtSensor.mjSENS_TORQUE,
                    objtype=mujoco.mjtObj.mjOBJ_SITE, objname="ft")

    # Torso: heightfield solo grafico (nessuna collisione).
    # COSA: aggiunge un heightfield (griglia di quote) di NR x NC punti, con size = [SX, SY, ELEV, BASE].
    hf = spec.add_hfield(name="tissue_visual", nrow=NR, ncol=NC, size=[SX, SY, ELEV, BASE])
    # COSA: quote iniziali: 4941 valori tutti 0.4 -> quota 0.4 * ELEV = 0.10 m (piatto all'altezza dell'apice H).
    # PERCHÉ: MuJoCo vuole dei dati per compilare; poi update_tissue_mesh() li sovrascrive a ogni fotogramma.
    #         Senza viewer (simulazione batch) resta piatto: non importa, e' solo grafica.
    hf.userdata = np.full(NR * NC, 0.4)
    # COSA: la geometria che DISEGNA l'heightfield, attaccata al mondo (fissa); contype=0, conaffinity=0: nessuna
    #       collisione, quindi MuJoCo non calcola contatti fra sonda e torso. Colore pelle.
    # PERCHÉ: il contatto lo calcola Tissue.contact() (Hunt-Crossley) e lo applica apply_tip_force().
    spec.worldbody.add_geom(name="tissue_visual_geom", type=mujoco.mjtGeom.mjGEOM_HFIELD,
                            hfieldname="tissue_visual", contype=0, conaffinity=0,
                            rgba=[0.80, 0.55, 0.45, 1])
    # COSA: pavimento: piano 2 m x 2 m (size = semi-lati 1, 1; il terzo numero e' il passo della griglia disegnata),
    #       a quota -BASE - 0.003 = -0.053 m, appena sotto il piedistallo dell'heightfield. Senza collisioni.
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1.0, 1.0, 0.05],
                            contype=0, conaffinity=0, rgba=[0.25, 0.3, 0.35, 1], pos=[0, 0, -BASE - 0.003])
    # COSA: luce direzionale dall'alto (dir = -z), diffusa al 45%, niente riflessi speculari. Solo grafica.
    spec.worldbody.add_light(pos=[0, 0, 1.5], dir=[0, 0, -1],
                             type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                             diffuse=[0.45, 0.45, 0.45], specular=[0, 0, 0])

    # COSA: aggiunge al modello un keyframe (una "fotografia" dello stato) di nome "hover": qpos = Q_HOVER,
    #       qvel = 0, ctrl = 0. Si affianca al keyframe "home" che arriva dal file XML delle Menagerie.
    # PERCHÉ: la posa di partenza sta DENTRO il modello: si carica con una riga, senza calcoli.
    # DOVE: AdmittanceSim.__init__: mujoco.mj_resetDataKeyframe(m, d, m.key("hover").id).
    spec.add_key(name="hover", qpos=Q_HOVER)

    # COSA: compile() trasforma la descrizione (MjSpec) nel modello numerico MjModel: array di numeri fissi
    #       (masse, geometrie, indirizzi). Da qui la STRUTTURA non cambia piu', solo i valori negli array.
    #       Misurato: nq = nv = nu = 7, 11 corpi, 64 geometrie, 3 siti, 2 sensori.
    model = spec.compile()
    # COSA: passo di integrazione 1 ms (il file XML usa il default di MuJoCo, 0.002 s, misurato).
    # PERCHÉ: 1 kHz e' la frequenza del controllo del Franka vero e della lettura dell'ATI: un passo fisico =
    #         un giro del controllore.
    # DOVE: AdmittanceSim.dt = m.opt.timestep -> ammettenza (Eulero), traiettoria, q_ref += q_dot*dt, sensore.
    model.opt.timestep = 0.001
    # COSA: mj_name2id(modello, tipo, nome) restituisce l'INDICE intero di un oggetto dal suo nome. dict(...) crea
    #       un dizionario chiave -> indice. Misurato: site 1, tip_geom 63, hfield 0, probe_body 10, ft_site 2, ft_adr 0.
    # PERCHÉ: gli array di MuJoCo (site_xpos, xfrc_applied, geom_rgba, ...) si indicizzano con numeri; cercare
    #         il nome a ogni millisecondo sarebbe lento e verboso: lo si fa una volta qui.
    # DOVE: AdmittanceSim.ids, usato in __init__, step() e run_viewer.
    ids = dict(
        # sito della punta: posizione/orientamento/Jacobiano del punto controllato
        site=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tip"),
        # sfera verde: colore e centro per la grafica
        tip_geom=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "tip_vis"),
        # heightfield: run_viewer chiama v.update_hfield(ids["hfield"]) per mandarlo alla scheda grafica
        hfield=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, "tissue_visual"),
        # corpo della sonda: riga di d.xfrc_applied su cui apply_tip_force scrive la forza del tessuto
        probe_body=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "probe"),
        # sito del sensore: il suo orientamento R_ft porta la forza misurata nel mondo (step, BLOCCO 0)
        ft_site=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ft"),
        # COSA: sensor_adr[id] = posizione in d.sensordata da cui partono i numeri di quel sensore. int(...) toglie
        #       il tipo numpy. DOVE: step() legge d.sensordata[ft_adr : ft_adr + 6] = wrench vero.
        ft_adr=int(model.sensor_adr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "ft_force")]),
    )
    # forza (3) e coppia (3) devono essere contigue in sensordata: ft_adr .. ft_adr+6
    # COSA: in step() taglio 6 numeri da sensordata: d.sensordata[ft_adr : ft_adr + 6]. La fetta da' SEMPRE 6 numeri,
    #       qualunque cosa ci sia dentro. I posti 1-3 sono la forza di sicuro (la fetta parte da ft_adr).
    #       L'assert controlla che al 4° POSTO della fetta cominci la COPPIA, e quindi che 4°, 5° e 6° siano coppia:
    #       sensor_adr[ft_torque] == ft_adr + 3. Oggi 3 == 0 + 3 -> vero, passa in silenzio.
    #       Se e' falso -> AssertionError e il programma si ferma QUI.
    # PERCHÉ: se un giorno metti un altro sensore fra forza e coppia (es. un accelerometro), sensordata diventa
    #         [Fx Fy Fz ax ay az Tx Ty Tz]: la fetta resta lunga 6, ma i posti 4-6 sono accelerazioni e l'ATI le
    #         prenderebbe per coppie senza nessun errore. L'assert lo impedisce.
    assert model.sensor_adr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "ft_torque")] == ids["ft_adr"] + 3
    # DOVE: AdmittanceSim.__init__: self.model, self.ids = ps.build_model()
    return model, ids


# ------------------------------------------------------------------------------------- cinematica
def rot_err(R, Rd):
    """Errore di orientamento (vettore 3D, piccolo angolo) che porta R verso Rd."""
    # COSA: R e Rd sono matrici 3 x 3 con TRE colonne (0, 1, 2). La colonna i e' l'asse i (x, y, z) del frame
    #       della sonda scritto nel mondo: R[:, i] = dove punta ORA, Rd[:, i] = dove DOVREBBE puntare.
    #       1) Prodotto vettoriale di due versori a x b: e' perpendicolare a entrambi, quindi e' l'ASSE attorno a
    #          cui ruotare a per portarlo su b (mano destra), e ha lunghezza sin(angolo fra a e b).
    #       2) Se per passare da R a Rd serve una rotazione di theta attorno all'asse u (versore, nel mondo):
    #          - l'asse della sonda parallelo a u non si muove        -> R_i x Rd_i = 0
    #          - i due assi perpendicolari a u ruotano di theta        -> ognuno da' sin(theta) * u
    #          somma dei tre = 2 sin(theta) * u, quindi
    #              0.5 * (R_0 x Rd_0 + R_1 x Rd_1 + R_2 x Rd_2) = sin(theta) * u
    #          (identita' ESATTA per qualunque u, non solo per u allineato a un asse).
    #          Forma matriciale equivalente: Q = Rd R^T (rotazione d'errore), risultato = vee((Q - Q^T) / 2).
    #       3) RISULTATO: un VETTORE di 3 numeri (non uno scalare). Direzione = asse di rotazione u,
    #          lunghezza = sin(theta) ~ theta [rad] per angoli piccoli: e' il "vettore di rotazione" R -> Rd.
    #       Esempio: R = I, Rd = theta attorno a z: x x (cos, sin, 0) = (0, 0, sin); y x (-sin, cos, 0) = (0, 0, sin);
    #       z x z = 0 -> 0.5 * somma = (0, 0, sin theta). Misurato: 10 gradi attorno a x -> (0.1736, 0, 0).
    # PERCHÉ: serve un errore di orientamento a 3 numeri, da trasformare in velocita' angolare con un guadagno P:
    #         w = KP_ROT_V * errore. KP_ROT_V = 4 [1/s] NON e' uno smorzamento: e' un guadagno proporzionale, cioe'
    #         il fattore che converte l'errore in radianti in una velocita' angolare in rad/s: per ogni radiante di
    #         errore comanda 4 rad/s (es. errore 0.1 rad -> w = 0.4 rad/s). Unita': [rad] * [1/s] = [rad/s].
    #         Piu' l'errore cala, piu' w cala: la sonda si riallinea dolcemente. Se il robot insegue bene,
    #         l'errore decade come exp(-4 t) (costante di tempo tau = 1/4 = 0.25 s).
    #         Limite: sin(theta) cresce solo fino a 90 gradi, poi cala, e a 180 gradi vale 0 -> va bene solo per
    #         errori piccoli (il nostro caso: la sonda resta quasi verticale).
    # DOVE: step() BLOCCO 4: w = clip(KP_ROT_V * rot_err(R, R_des), +-W_MAX) tiene la sonda verticale; w va in
    #       v6[3:] -> differential_ik.
    return 0.5 * (np.cross(R[:, 0], Rd[:, 0]) + np.cross(R[:, 1], Rd[:, 1]) + np.cross(R[:, 2], Rd[:, 2]))


def jac(model, data, site):
    """Jacobiano 6 x nv del sito: righe 0-2 velocita' lineare, righe 3-5 velocita' angolare."""
    # COSA: due matrici di zeri 3 x nv (nv = 7 giunti), da riempire: jp = parte di posizione, jr = di rotazione.
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    # COSA: MuJoCo riempie jp e jr DENTRO gli array passati (non restituisce niente):
    #       jp[:, j] = quanto si sposta il sito (m) per 1 rad del giunto j; jr[:, j] = quanto ruota (rad).
    #       Usa le posizioni gia' calcolate in data (serve prima mj_forward o mj_step1).
    mujoco.mj_jacSite(model, data, jp, jr, site)
    # COSA: impila le due matrici una sopra l'altra -> J 6 x 7, tale che [v_lineare; w_angolare] = J @ q_dot.
    #       Misurato alla posa home, colonna 0 = (0, 0.554, 0 | 0, 0, 1): il giunto 1 ruota attorno a z del mondo
    #       e la punta, a 0.554 m dall'asse, si muove lungo y di 0.554 m/s per ogni rad/s.
    # PERCHÉ: e' il "ponte" fra spazio dei giunti e spazio cartesiano della punta.
    # DOVE: step() BLOCCO 0 (v_tip = J @ qd, velocita' della punta per lo smorzamento di Hunt-Crossley)
    #       e BLOCCO 5 (differential_ik: da velocita' cartesiana v6 a velocita' dei giunti).
    return np.vstack([jp, jr])


# ------------------------------------------------------------------------------------- forza del tessuto
def apply_tip_force(data, body, force, point):
    """
    Applica la forza del tessuto `force` (mondo) nel punto `point` del corpo `body` tramite xfrc_applied:
    forza + coppia rispetto al baricentro del corpo, in mondo. A differenza di qfrc_applied, xfrc_applied
    e' visto anche dai sensori force/torque di MuJoCo (il sensore ATI alla flangia).
    """
    # IDEA GENERALE: CAMBIO DI POLO (punta -> baricentro), non cambio della fisica.
    #   - Noi abbiamo tutto calcolato sulla PUNTA della sonda: Tissue.contact() da' la forza F che il tessuto
    #     applica nel punto "tip" (point). Riferita alla punta, l'azione del tessuto e' (forza F, coppia 0).
    #   - MuJoCo invece vuole il wrench esterno di un corpo riferito al suo BARICENTRO (d.xipos[body]),
    #     con componenti nel frame MONDO: non accetta "una forza in un punto qualunque".
    #   - Quindi spostiamo il punto di riferimento (il POLO dei momenti) dalla punta al baricentro. Regola dei
    #     corpi rigidi: spostando il polo la forza resta IDENTICA, la coppia diventa r x F (r = punta - baricentro):
    #         riferita alla punta:      forza F,  coppia 0
    #         riferita al baricentro:   forza F,  coppia r x F      <- le due descrizioni sono EQUIVALENTI
    #     Il corpo riceve la stessa spinta e la stessa tendenza a ruotare: cambia solo come la descriviamo.
    #   - NB: e' un cambio di POLO, non di frame: gli assi restano quelli del mondo.
    #   Risultato: la riga "probe" di xfrc_applied = wrench a 6 numeri [Fx, Fy, Fz, tx, ty, tz] = [F, r x F].
    #
    # COSA: d.xfrc_applied e' una tabella (n. corpi x 6): per ogni corpo [fx, fy, fz, tx, ty, tz] nel MONDO,
    #       riferiti al BARICENTRO del corpo. Le prime 3 colonne della riga "probe" = forza del tessuto, uguale
    #       a quella calcolata sulla punta (spostando il polo la forza non cambia).
    #       E' un'ASSEGNAZIONE: a ogni passo la forza vecchia viene sostituita, non sommata (MuJoCo non la azzera
    #       da solo: resta li' finche' non la riscriviamo).
    data.xfrc_applied[body, :3] = force
    # COSA: le ultime 3 colonne = la coppia dovuta al cambio di polo: tau = r x F, con
    #       r = point - d.xipos[body] = punta - baricentro (nel mondo). E' l'UNICA coppia: il tessuto spinge solo
    #       con una forza in un punto, quindi la coppia nasce solo dallo spostamento del polo.
    #       Se F e' parallela a r (pelle piatta, forza verticale) r x F = 0: niente coppia.
    #       Misurato a regime (partenza serpentina, pelle inclinata): forza (0, -1.56, 4.75) N, r = (0, 0, -0.075) m
    #       -> coppia (-0.117, 0, 0) Nm.
    # PERCHÉ: senza questa coppia la fisica cambierebbe davvero: sarebbe come spingere la sonda nel baricentro,
    #         7.5 cm piu' su della punta. Il robot si muoverebbe in modo diverso e il sensore alla flangia
    #         leggerebbe una coppia sbagliata.
    # DOVE: chiamata in step() BLOCCO 0, ogni ms, PRIMA di mj_step2, che la usa per muovere il robot e per
    #       calcolare il sensore force/torque.
    data.xfrc_applied[body, 3:] = np.cross(point - data.xipos[body], force)


# ------------------------------------------------------------------------------------- grafica del torso
# IDEA GENERALE: l'heightfield e' una griglia di 61 x 81 punti e per OGNI punto dobbiamo dare a MuJoCo una quota z.
#   Per calcolare la quota di un punto (z0(x, y), surface_height(x, y, t)) bisogna sapere DOVE sta: la sua (x, y).
#   Qui prepariamo, una volta per tutte, le coordinate (x, y) di tutti i 4941 punti della griglia.
#
# PASSO 1 - i due "righelli" sui bordi della griglia. linspace(a, b, n) = n numeri equidistanti da a a b:
#   _gx: 81 valori di x da -SX = -0.20 a +SX = +0.20 m  (passo 0.40 / 80 = 5 mm)  -> dove cadono le COLONNE
#   _gy: 61 valori di y da -SY = -0.15 a +SY = +0.15 m  (passo 0.30 / 60 = 5 mm)  -> dove cadono le RIGHE
#   Estremi = gli stessi SX, SY passati ad add_hfield: MuJoCo distribuisce i punti dell'heightfield proprio
#   fra -SX..+SX e -SY..+SY, cosi' ogni quota calcolata cade sul punto giusto.
_gx = np.linspace(-SX, SX, NC)
_gy = np.linspace(-SY, SY, NR)
# PASSO 2 - da due righelli alle coordinate di OGNI incrocio. meshgrid crea due matrici 61 x 81 (come la griglia):
#   _GX = la x di ogni punto: valori uguali dentro ogni colonna -> _GX[i, j] = _gx[j]  (dipende solo dalla colonna j)
#   _GY = la y di ogni punto: valori uguali dentro ogni riga    -> _GY[i, j] = _gy[i]  (dipende solo dalla riga i)
#   Il punto (riga i, colonna j) della griglia ha coordinate (_GX[i, j], _GY[i, j]).
#   ESEMPIO PICCOLO: 3 colonne _gx = [-1, 0, 1], 2 righe _gy = [-1, 1]. I punti della griglia (incroci
#   fra righe e colonne), con le coordinate (x, y) di ognuno:
#                     x = -1     x = 0     x = 1
#         y = -1  ->  (-1,-1)    ( 0,-1)    ( 1,-1)      <- riga 0
#         y =  1  ->  (-1, 1)    ( 0, 1)    ( 1, 1)      <- riga 1
#   numpy non mette una coppia in una casella: meshgrid divide questa tabella in DUE tabelle con la stessa forma,
#   una con solo la x di ogni punto e una con solo la y:
#         _GX (la x di ogni punto)      _GY (la y di ogni punto)
#            -1    0    1                  -1   -1   -1
#            -1    0    1                   1    1    1
#   In _GX i valori DENTRO ogni COLONNA sono uguali (una colonna = una x fissa: -1,-1 | 0,0 | 1,1): la x cambia
#   solo andando verso destra. In _GY i valori DENTRO ogni RIGA sono uguali (una riga = una y fissa: -1,-1,-1 |
#   1,1,1): la y cambia solo scendendo. Punto in basso a destra: _GX = 1, _GY = 1 -> e' il punto (1, 1).
#   Poi z0(_GX, _GY) calcola casella per casella e restituisce una terza tabella con la QUOTA di ogni punto:
#         z(-1,-1)   z(0,-1)   z(1,-1)
#         z(-1, 1)   z(0, 1)   z(1, 1)
#   Nel codice e' identico ma piu' grande: 61 righe (le y) x 81 colonne (le x) = 4941 punti.
# PERCHÉ:
#   - UN calcolo invece di un doppio ciclo: z0(_GX, _GY) applica la formula casella per casella e restituisce
#     direttamente la matrice 61 x 81 delle quote. Senza meshgrid servirebbero due for annidati
#     (for i in 61: for j in 81: z[i, j] = z0(_gx[j], _gy[i])) = 4941 chiamate Python a ogni fotogramma.
#   - righe = y, colonne = x e' l'ordine di MuJoCo (prima riga a y = -SY, prima colonna a x = -SX): cosi' in
#     update_tissue_mesh ravel() (appiattisce riga per riga) da' i valori gia' nell'ordine di hfield_data.
#   - Calcolate UNA volta sola, all'import del modulo: le coordinate dei punti non cambiano mai (la griglia e'
#     fissa), cambiano solo le quote. Il trattino basso _ = convenzione Python per "uso interno a questo file".
# DOVE: update_tissue_mesh (z0, surface_height e avvallamento gaussiano su tutta la griglia).
_GX, _GY = np.meshgrid(_gx, _gy)        # shape (NR, NC) = (61, 81): righe = y, colonne = x


def update_tissue_mesh(model, tissue, t, vis_gain=1.0, deformation=None):
    """
    Aggiorna la grafica del torso: superficie al tempo t (respiro esagerato di vis_gain) e,
    se deformation = (x, y, penetrazione) [m], l'avvallamento sotto la sonda.
    """
    # COSA: forma a riposo del torso (ellisse, TorsoGeometry.z0) su tutta la griglia: matrice 61 x 81.
    z0 = tissue.geom.z0(_GX, _GY)
    # COSA: surface_height = forma + respiro + sussulti; togliendo z0 resta solo il MOTO della pelle, che viene
    #       moltiplicato per vis_gain. Con vis_gain = 1 (default) z = surface_height esatta.
    # PERCHÉ: il respiro e' di pochi mm, a schermo quasi invisibile: --vis-gain 5 lo esagera SOLO nel disegno.
    z = z0 + vis_gain * (tissue.surface_height(_GX, _GY, t) - z0)
    # COSA: se il chiamante passa una deformazione (cioe' la sfera e' dentro la pelle) disegno l'avvallamento.
    #       CHI LA FORNISCE: run_viewer (run_admittance.py), a ogni fotogramma: comp = measure_tissue_compression(...),
    #       poi deformation = (x centro sfera, y centro sfera, comp) se comp > 0, altrimenti None.
    #       Non viene dal controllore ne' da tissue: esiste solo quando c'e' il viewer.
    if deformation is not None:
        # COSA: spacchetta la tupla: centro (x, y) della sfera e quanto e' affondata [m].
        x, y, penetration = deformation
        # COSA: profondita' fra 0 e 15 mm (sicurezza grafica: niente buchi assurdi).
        depth = float(np.clip(penetration, 0.0, 0.015))
        # COSA: sotto 0.01 mm non disegna nulla (avvallamento invisibile, inutile calcolarlo).
        if depth > 1e-5:
            # COSA: raggio dell'impronta di una sfera di raggio R affondata di depth in un piano:
            #       a^2 = 2 R depth - depth^2 ~ 2 R depth (teorema delle corde). R = 22 mm, depth = 6.4 mm -> a ~ 17 mm.
            radius = np.sqrt(2.0 * TIP_RADIUS * depth)
            # COSA: larghezza della campana fra 12 mm (almeno ~2 celle da 5 mm, altrimenti non si vede) e 35 mm.
            sigma = float(np.clip(radius, 0.012, 0.035))
            # COSA: sottrae una campana gaussiana 2D centrata sotto la sonda: profondita' depth al centro,
            #       exp(-0.5 * ((X-x)^2 + (Y-y)^2) / sigma^2) va da 1 al centro a ~0 a 3 sigma.
            z = z - depth * np.exp(-0.5 * (((_GX - x) / sigma) ** 2 + ((_GY - y) / sigma) ** 2))
    # COSA: indice dell'heightfield "tissue_visual" nel modello compilato (0).
    hid = model.hfield("tissue_visual").id
    # COSA: start e' un NUMERO intero, non un array. MuJoCo tiene le quote di TUTTI gli heightfield del modello una
    #       dopo l'altra in UN array lungo, model.hfield_data; hfield_adr[hid] = posizione da cui partono le quote
    #       del nostro. Qui c'e' un solo heightfield -> start = 0, e le nostre quote sono hfield_data[0 : 4941], da dove parte il posto per hfield
    start = int(model.hfield_adr[hid])
    # DOVE STA hfield_data (riferimento teorico):
    #   - E' un campo dell'oggetto MODELLO (MjModel, uscito da spec.compile()), in RAM, dentro la struttura C di
    #     MuJoCo. Da Python e' un array numpy che PUNTA a quella memoria: scrivendolo modifichi il modello vero,
    #     non una copia. MjModel = descrizione del sistema (geometrie, masse, heightfield; di solito fissa),
    #     MjData = stato che cambia a ogni passo (qpos, qvel, xfrc_applied, sensordata). Le quote sono geometria
    #     -> stanno nel modello.
    #   - CONTIENE solo quote: quelle di TUTTI gli heightfield del modello, una dopo l'altra:
    #         hfield_data = [ quote hfield 0 | quote hfield 1 | ... ]   (ognuno parte da hfield_adr[i])
    #     Le altre informazioni di ogni heightfield stanno in array separati: hfield_nrow, hfield_ncol (griglia),
    #     hfield_size (SX, SY, ELEV, BASE), hfield_adr (inizio delle sue quote). Noi ne abbiamo UNO solo ->
    #     hfield_data e' lungo esattamente 4941 = 61 x 81 e lo riscriviamo tutto.
    #   - CHI LO LEGGE: il viewer, ma NON direttamente: all'avvio la scheda grafica si copia la superficie e da
    #     li' disegna la sua copia. Per questo run_viewer, dopo questa funzione, chiama v.update_hfield(id):
    #     rimanda le quote nuove alla scheda grafica (senza, in RAM la pelle cambia ma a schermo resta quella
    #     vecchia). La FISICA non lo legge mai: la geom ha contype = conaffinity = 0 (niente collisioni); con le
    #     collisioni attive sarebbe il calcolo dei contatti di MuJoCo a leggere queste quote.
    # COSA: MuJoCo vuole quote normalizzate fra 0 e 1 (quota vera = dato * ELEV): z / ELEV, tagliato in [0, 1].
    #       ravel() appiattisce la matrice 61 x 81 riga per riga (riga = una y), l'ordine che si aspetta MuJoCo.
    #       Si scrive nel MODELLO (non in data): e' geometria. La scheda grafica lo riceve con v.update_hfield().
    # DOVE: run_viewer, a ogni fotogramma (16 ms di simulazione). Non tocca la fisica.
    model.hfield_data[start:start + NR * NC] = np.clip(z / ELEV, 0.0, 1.0).ravel()


def measure_tissue_compression(model, data, ids, tissue, t):
    """Compressione geometrica sfera-superficie (centro e raggio della sfera): serve solo alla grafica."""
    # COSA: centro della sfera verde nel mondo (3 numeri).
    center = data.geom_xpos[ids["tip_geom"]]
    # COSA: le sue x e y come float Python.
    x, y = float(center[0]), float(center[1])
    # COSA: punto della pelle proprio sotto il centro (stessa x, y; quota della superficie al tempo t).
    surface = np.array([x, y, tissue.surface_height(x, y, t)])
    # COSA: normale uscente della pelle in quel punto (Tissue.normal, vettore di lunghezza 1).
    normal = tissue.normal(x, y, t)
    # COSA: raggio della sfera letto dal modello (geom_size[..., 0] = 0.022).
    radius = float(model.geom_size[ids["tip_geom"], 0])
    # COSA: distanza CON SEGNO del centro dal piano tangente alla pelle, lungo la normale (> 0 = centro sopra).
    signed_distance = float(np.dot(center - surface, normal))
    # COSA: quanto la sfera entra nel piano: raggio - distanza del centro; 0 se non tocca.
    #       NON e' il delta della fisica (Tissue.contact usa il solo punto "tip", fondo della sfera). Misurato:
    #       pelle piatta in (0, 0): 6.354 mm contro delta 6.346. Pelle inclinata, partenza della
    #       serpentina a t = 6 s: 9.10 contro 7.99 mm. Su una pelle inclinata di theta il punto piu' profondo della
    #       sfera non e' il suo fondo: differenza R (1 - cos theta) = 22 mm * (1 - 0.95) = 1.1 mm.
    # DOVE: run_viewer -> update_tissue_mesh(deformation = (x, y, compressione)): solo per disegnare l'avvallamento.
    return max(0.0, radius - signed_distance)
