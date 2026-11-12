"""
Sensore forza/coppia a 6 assi ATI Nano17 al polso del Panda (flangia -> sensore -> sonda), con box Net F/T.

Il sensore e' modellato a partire dal wrench VERO misurato da MuJoCo (sensori `force` e `torque` sul sito "ft"
della flangia). Quel wrench contiene tutto cio' che un sensore reale vede: forza del tessuto, peso della sonda e
forze d'inerzia della sonda quando accelera.

Catena di misura (per ogni asse, una lettura per passo di controllo da 1 ms):

    wrench vero -> + rumore -> filtro passa-basso Net F/T -> saturazione alla portata -> ritardo -> - tara

Dati dalle specifiche ATI (Nano17, sito ATI e manuale Net F/T 9620-05-NET FT-23):
  * portata e risoluzione delle calibrazioni SI-12-0.12, SI-25-0.25, SI-50-0.5 (tabella CALIBRATIONS);
    la risoluzione dichiarata e' "effective resolution after dropping four counts of noise";
  * filtri Net F/T selezionabili: nessuno, 838, 326, 152, 73, 35, 18, 8, 5 Hz. Il ritardo di gruppo dei filtri
    (manuale, fig. 18.2) coincide con quello di un passa-basso del primo ordine, 1/(2*pi*fc): 35 Hz -> ~4.5 ms;
  * latenza senza filtro fino all'Ethernet: 286 us.

IPOTESI DA MISURARE IN LABORATORIO (non sono specifiche ATI):
  * deviazione standard del rumore = risoluzione dichiarata (noise_scale = 1);
  * ritardo di trasporto (rete, PC) = 0 campioni oltre a quello intrinseco della simulazione (1 ms, vedi sotto);
  * deriva termica della tara = 0.
"""
from collections import deque
import numpy as np

# portate e risoluzioni Nano17 (sito ATI): (Fx, Fy, Fz [N], Tx, Ty, Tz [Nm]), risoluzione (forza [N], coppia [Nm])
CALIBRATIONS = {
    "SI-12-0.12": dict(range=(12, 12, 17, 0.12, 0.12, 0.12), res=(1 / 320, 1 / 64 * 1e-3)),
    "SI-25-0.25": dict(range=(25, 25, 35, 0.25, 0.25, 0.25), res=(1 / 160, 1 / 32 * 1e-3)),
    "SI-50-0.5": dict(range=(50, 50, 70, 0.50, 0.50, 0.50), res=(1 / 80, 1 / 16 * 1e-3)),
}

# frequenze di taglio selezionabili sul Net F/T [Hz]; None = "No Filter"
NETFT_FILTERS = (None, 838, 326, 152, 73, 35, 18, 8, 5)


class ATINanoSensor:
    """
    read(wrench_true)  -> wrench misurato nel frame del sensore, dopo la tara [Fx, Fy, Fz, Tx, Ty, Tz]
    tare(wrench_true)  -> "Bias" del Net F/T: memorizza l'offset attuale (media di n letture rumorose)

    wrench_true: [forza (3), coppia (3)] nel frame del sito "ft", come lo restituisce MuJoCo.
    """

    def __init__(self, dt, calibration="SI-25-0.25", filter_hz=None, noise_scale=1.0,
                 transport_delay_steps=0, drift_per_s=0.0, seed=0):
        if calibration not in CALIBRATIONS:   #controllo che la stringa sia una chiave valida del dizionario CALIBRATIONS
            raise ValueError(f"calibrazione sconosciuta {calibration}; disponibili: {list(CALIBRATIONS)}")
        if filter_hz not in NETFT_FILTERS:      #controllo che la frequenza di taglio sia tra quelle disponibili sul Net F/T
            raise ValueError(f"filtro non disponibile sul Net F/T: {filter_hz}; disponibili: {NETFT_FILTERS}")
        cal = CALIBRATIONS[calibration]
        self.dt = dt
        self.calibration, self.filter_hz = calibration, filter_hz
        self.range = np.array(cal["range"], dtype=float)  #trasformo tupla portate in array numpy
        res_f, res_t = cal["res"]
        self.resolution = np.array([res_f] * 3 + [res_t] * 3) #sigma di quantizzazione (risoluzione) per ogni asse, in N o Nm quindi 2 vettori da 3 elementi concatenati
        self.noise_std = noise_scale * self.resolution  #scala del rumore per il rumore gaussiano (deviazione standard) = risoluzione dichiarata * fattore di scala
        # filtro del primo ordine, discretizzazione esatta; senza filtro a = 1 (passa tutto)
        self.a = 1.0 if filter_hz is None else 1.0 - np.exp(-2 * np.pi * filter_hz * dt)
        self.y = None                                   # stato del filtro (inizializzato alla prima lettura)
        self.buf = deque(maxlen=transport_delay_steps + 1) # buffer per il ritardo di trasporto (rete, PC) ha solo un elemento se transport_delay_steps=0, altrimenti contiene i campioni ritardati
        self.offset = np.zeros(6)                       # tara ("software bias vector")
        self.drift = np.array([drift_per_s] * 3 + [0.0] * 3)
        self.rng = np.random.default_rng(seed)  #generatore di numeri casuali per il rumore gaussiano
        self.saturated = False
    #Stato dell'oggetto alla fine del costruttore: portate caricate, σ del rumore 6.25 mN, a = 1, filtro vuoto, coda vuota di lunghezza 1, tara zero
    
    def _acquire(self, wrench_true):
        """Rumore -> filtro -> saturazione. Restituisce il campione grezzo (prima di ritardo e tara)."""
        raw = np.asarray(wrench_true, dtype=float) + self.rng.normal(0.0, self.noise_std)
        if self.y is None:
            self.y = raw.copy()
        else:
            self.y = self.y + self.a * (raw - self.y)   #filtro passa-basso del primo ordine (discretizzazione esatta)
        self.saturated = bool(np.any(np.abs(self.y) > self.range)) #almeno un asse fuori portata ho saturazione
        return np.clip(self.y, -self.range, self.range) #campione dopo rumore, filtro e saturazione (ritardo e tara non ancora applicati)

    def tare(self, wrench_true, n=100): #n è il numero di letture da mediare per fare la tara
        """Come il pulsante Bias del Net F/T: offset = media di n letture (rumorose) del carico attuale."""
        samples = [self._acquire(wrench_true) for _ in range(n)]  #salvo 100 campioni rumorosi del wrench vero
        self.offset = np.mean(samples, axis=0) #faccio la media per ottenere l'offset (tara) da sottrarre alle letture future
        self.buf.clear()

    def read(self, wrench_true):
        self.offset = self.offset - self.drift * self.dt  # deriva della tara (0 di default)
        self.buf.append(self._acquire(wrench_true))  #acquisisco il wrench vero, lo filtro, lo satura e lo metto nel buffer (ritardo)
        return self.buf[0] - self.offset   #campione in testa alla coda meno la tara: 6 numeri nel frame del sensore; admittance_control li porta nel mondo e ne ricava F_meas
    
