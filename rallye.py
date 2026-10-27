# Graph, der die Stationen und Zeiten dazwischen abbildet
# Gruppen haben eine feste Route und eine Startzeit
# Stationen haben eine Kapazität: ist sie voll, warten die Gruppen in einer Warteschlange
from __future__ import annotations
from pathlib import Path
from typing import Optional
from collections import defaultdict, deque
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.animation import FuncAnimation

# ----------------------------------------------------------------------
# Karte: Kanten = Reisedauer, Knoten = Aufenthaltsdauer (in Ticks)
# ----------------------------------------------------------------------
G = nx.Graph()
G.add_edge("Start", "Tanzen", dauer=40)
G.add_edge("Tanzen", "Pyramide", dauer=25)
G.add_edge("Tanzen", "Amazonas", dauer=27)
G.add_edge("Pyramide", "Amazonas", dauer=5)
G.add_edge("Amazonas", "Pantomime", dauer=7)
G.add_edge("Pantomime", "Parkour", dauer=30)
G.add_edge("Parkour", "Gegenstände", dauer=20)
G.add_edge("Gegenstände", "Ziel", dauer=25)
G.add_edge("Start", "Gegenstände", dauer=25)
G.add_edge("Pantomime", "Tanzen", dauer=6)
G.add_edge("Parkour", "Amazonas", dauer=35)
G.add_edge("Amazonas", "Ziel", dauer=35)
G.add_edge("Pyramide", "Ziel", dauer=43)
G.add_edge("Start","Pantomime", dauer=40)
G.add_edge("Tanzen","Parkour", dauer=18)
G.add_edge("Amazonas", "Start", dauer=35)
G.add_edge("Pantomime", "Ziel", dauer=43)

G.add_edge("Theather", "Ziel", dauer=20)
G.add_edge("Theather", "Amazonas", dauer=17)
G.add_edge("Theather", "Pyramide", dauer=13)

# Aufenthaltsdauer an den Stationen
G.nodes["Start"]["dauer"] = 0
G.nodes["Tanzen"]["dauer"] = 30
G.nodes["Pantomime"]["dauer"] = 9
G.nodes["Parkour"]["dauer"] = 11
G.nodes["Pyramide"]["dauer"] = 15
G.nodes["Amazonas"]["dauer"] = 10
G.nodes["Gegenstände"]["dauer"] = 15
G.nodes["Ziel"]["dauer"] = 0

#G.nodes["Theather"]["dauer"] = 0

# Kapazität = max. Gruppen gleichzeitig an der Station (mindestens 1!)
# Ohne Eintrag ist die Station unbegrenzt (z. B. Start, Ziel).
# Die Werte sind Platzhalter, bitte an deine Rallye anpassen.
G.nodes["Tanzen"]["kapazitaet"] = 4
G.nodes["Pantomime"]["kapazitaet"] = 2
G.nodes["Parkour"]["kapazitaet"] = 2
G.nodes["Pyramide"]["kapazitaet"] = 3
G.nodes["Amazonas"]["kapazitaet"] = 2
G.nodes["Gegenstände"]["kapazitaet"] = 3

#G.nodes["Theather"]["kapazitaet"] = 10

# Mindestabstand (in Ticks) zwischen zwei Gruppen, die dieselbe Station verlassen
ABGABE_ABSTAND = 3


# ----------------------------------------------------------------------
# Stationen: Belegung und Warteschlangen
# ----------------------------------------------------------------------
class Stationen:
    """Verwaltet Belegung und Warteschlangen aller Stationen."""

    def __init__(self, G):
        self.G = G
        self.belegt = defaultdict(int)
        self.queue = defaultdict(deque)
        self.letzte_abfahrt = defaultdict(lambda: -10**9)   # Tick der letzten Abgabe

    def _frei(self, k):
        kap = self.G.nodes[k].get("kapazitaet")   # None = unbegrenzt
        return kap is None or self.belegt[k] < kap

    def _aufnehmen(self, g):
        k = g.position
        self.belegt[k] += 1
        g.belegt = True
        g.wartet = False
        g.restzeit = self.G.nodes[k].get("dauer", 0)

    def ankunft(self, g):
        """Gruppe kommt an: entweder sofort rein oder in die Warteschlange."""
        k = g.position
        if self.G.nodes[k].get("dauer", 0) == 0:
            return                                  # kein Aufenthalt, keine Kapazität nötig
        if self._frei(k) and not self.queue[k]:
            self._aufnehmen(g)
        else:
            g.wartet = True
            self.queue[k].append(g)

    def abgabe_frei(self, k, tick):
        """Darf in diesem Tick eine Gruppe die Station k verlassen?"""
        return tick - self.letzte_abfahrt[k] >= ABGABE_ABSTAND

    def verlassen(self, g, tick):
        """Gruppe geht weiter: Platz frei, nächste aus der Warteschlange rückt nach."""
        if not g.belegt:
            return
        k = g.position
        g.belegt = False
        self.belegt[k] -= 1
        self.letzte_abfahrt[k] = tick
        while self.queue[k] and self._frei(k):
            self._aufnehmen(self.queue[k].popleft())


# ----------------------------------------------------------------------
# Gruppen
# ----------------------------------------------------------------------
@dataclass
class Gruppe:
    name: str
    route: list                    # feste Knotenfolge, z. B. ["A", "B", "C", "E"]
    startzeit: int = 0             # Tick, ab dem die Gruppe losgeht
    position: str = ""
    weg: list = field(default_factory=list)
    unterwegs_nach: Optional[str] = None
    restzeit: int = 0
    wartet: bool = False           # steht in der Warteschlange
    belegt: bool = False           # belegt gerade einen Platz an der Station
    wartezeit: int = 0             # Summe aller Warteticks

    def __post_init__(self):
        self.position = self.route[0]
        self.weg = list(self.route[1:])

    def pruefe_route(self, G):
        """Stellt sicher, dass alle Knoten und Verbindungen existieren."""
        for u, v in zip(self.route, self.route[1:]):
            if not G.has_edge(u, v):
                raise ValueError(f"{self.name}: Keine Kante zwischen {u} und {v}")
            

    def gestartet(self, tick):
        return tick >= self.startzeit

    def angekommen(self, tick):
        return (self.gestartet(tick) and not self.weg and not self.wartet
                and self.unterwegs_nach is None and self.restzeit == 0)

    def schritt(self, G, tick, stationen):
        if not self.gestartet(tick):
            return

        if self.wartet:
            self.wartezeit += 1
            return

        if self.restzeit > 0:
            self.restzeit -= 1
            if self.restzeit > 0:
                return

        # Reise beendet -> an der Station anmelden (Platz oder Warteschlange)
        if self.unterwegs_nach is not None:
            self.position = self.unterwegs_nach
            self.unterwegs_nach = None
            stationen.ankunft(self)
            if self.wartet or self.restzeit > 0:
                return

        # Aufenthalt vorbei -> ggf. warten, bis die Station wieder abgeben darf
        if self.belegt and not stationen.abgabe_frei(self.position, tick):
            self.wartezeit += 1
            return

        # Platz freigeben, nächste Kante starten
        stationen.verlassen(self, tick)
        if self.weg:
            naechster = self.weg.pop(0)
            self.unterwegs_nach = naechster
            self.restzeit = G[self.position][naechster]["dauer"]

    def status(self, tick):
        if not self.gestartet(tick):
            return f"wartet auf Start (Tick {self.startzeit})"
        if self.wartet:
            return f"WARTET vor {self.position} (Warteschlange)"
        if self.unterwegs_nach:
            return f"unterwegs {self.position} -> {self.unterwegs_nach} (noch {self.restzeit})"
        if self.restzeit > 0:
            return f"verweilt in {self.position} (noch {self.restzeit})"
        if not self.weg:
            return f"am Ziel {self.position}"
        return f"in {self.position}"


# ----------------------------------------------------------------------
# Routen und Gruppen
# ----------------------------------------------------------------------
ROUTE_S = ["Start", "Gegenstände", "Parkour", "Pantomime", "Tanzen","Pyramide", "Amazonas", "Ziel"]
ROUTE_K = ["Start", "Gegenstände", "Parkour", "Amazonas", "Pantomime", "Tanzen", "Ziel"]
ROUTE_MINUS_S = ["Start", "Gegenstände", "Parkour", "Amazonas", "Tanzen", "Pantomime", "Ziel"]

ROUTE_M = ["Start", "Tanzen","Pyramide", "Amazonas", "Pantomime", "Parkour", "Gegenstände", "Ziel"]
ROUTE_L = ["Start", "Amazonas","Pyramide","Tanzen","Pantomime","Parkour","Gegenstände","Ziel"]

ROUTE_D1_S = ["Start", "Gegenstände", "Parkour", "Pantomime", "Tanzen", "Amazonas", "Ziel"]
ROUTE_D2_S = ["Start", "Gegenstände", "Parkour", "Amazonas", "Tanzen", "Pantomime", "Ziel"]

ROUTE_D1_M = ["Start", "Tanzen", "Amazonas", "Pantomime", "Parkour", "Gegenstände", "Ziel"]
ROUTE_D2_M = ["Start", "Pantomime","Amazonas","Tanzen","Parkour","Gegenstände","Ziel"]


ROUTE_D1_SP = ["Start", "Gegenstände", "Parkour", "Pantomime", "Tanzen","Pyramide", "Amazonas", "Ziel"]
ROUTE_D2_SP = ["Start", "Gegenstände", "Parkour", "Amazonas","Pyramide", "Tanzen", "Pantomime", "Ziel"]

ROUTE_D1_MP = ["Start", "Tanzen","Pyramide", "Amazonas", "Pantomime", "Parkour", "Gegenstände", "Ziel"]
ROUTE_D2_MP = ["Start", "Pantomime","Amazonas","Pyramide","Tanzen","Parkour","Gegenstände","Ziel"]

gruppen = []
k = 0
n = 20
for i in range(n):
    if(k%2 == 0):
        gruppen.append(Gruppe(f"S{i+1}", route=ROUTE_D1_S, startzeit=i * 10 + 5))
        gruppen.append(Gruppe(f"M{i+1}", route=ROUTE_D1_M, startzeit=i * 10))
        k += 1
    else:
        gruppen.append(Gruppe(f"K{i+1}", route=ROUTE_D2_S, startzeit=i * 10 + 5))
        gruppen.append(Gruppe(f"D{i+1}", route=ROUTE_D2_M, startzeit=i * 10))
        k += 1


#gruppen = [
#    Gruppe("S1", route=ROUTE_S, startzeit=0),
#    Gruppe("M1", route=ROUTE_M, startzeit=5),
#    Gruppe("K1", route=ROUTE_K, startzeit=10),
#    Gruppe("S2", route=ROUTE_S, startzeit=15),
#    Gruppe("M2", route=ROUTE_M, startzeit=20),
#    Gruppe("K2", route=ROUTE_K, startzeit=25),
#    Gruppe("S3", route=ROUTE_S, startzeit=30),
#    Gruppe("M3", route=ROUTE_M, startzeit=35),
#    Gruppe("K3", route=ROUTE_K, startzeit=40),
#    Gruppe("S4", route=ROUTE_S, startzeit=45),
#    Gruppe("M4", route=ROUTE_M, startzeit=50),
#    Gruppe("K4", route=ROUTE_K, startzeit=55),
#    Gruppe("S5", route=ROUTE_S, startzeit=60),
#    Gruppe("M5", route=ROUTE_M, startzeit=65),
#    Gruppe("K5", route=ROUTE_K, startzeit=70),
#]
for g in gruppen:
    g.pruefe_route(G)

# ----------------------------------------------------------------------
# Layout der Karte
# ----------------------------------------------------------------------
# Kamada-Kawai nutzt die Dauer als Abstand: lange Wege sehen auch lang aus.
pos = nx.kamada_kawai_layout(G, weight="dauer")
pos = {k: np.array(v) for k, v in pos.items()}
# Alternativ eigene Koordinaten (z. B. nach echtem Stadtplan):
# pos = {"Start": np.array([0, 0]), "Tanzen": np.array([2, 1]), ...}


def koordinate(g, index, anzahl):
    """Aktuelle Zeichenposition einer Gruppe (Knoten oder auf der Kante)."""
    if g.unterwegs_nach:
        u, v = g.position, g.unterwegs_nach
        d = max(G[u][v]["dauer"], 1)
        f = 1 - g.restzeit / d
        p = pos[u] + (pos[v] - pos[u]) * f
    else:
        p = pos[g.position].copy()
    # kleiner Versatz, damit Gruppen am selben Ort nicht exakt übereinander liegen
    winkel = 2 * np.pi * index / anzahl
    return p + 0.04 * np.array([np.cos(winkel), np.sin(winkel)])


# ----------------------------------------------------------------------
# Simulation aufzeichnen
# ----------------------------------------------------------------------
bilder = []   # pro Tick: Liste von (x, y, gestartet, wartet)


def aufnehmen(tick):
    bilder.append([(*koordinate(g, i, len(gruppen)), g.gestartet(tick), g.wartet)
                   for i, g in enumerate(gruppen)])


stationen = Stationen(G)

# Aufzeichnung der Auslastung pro Tick
stationen_liste = [n for n in G.nodes if G.nodes[n].get("kapazitaet") is not None]
belegung_verlauf = {s: [] for s in stationen_liste}
schlange_verlauf = {s: [] for s in stationen_liste}

def messen():
    for s in stationen_liste:
        belegung_verlauf[s].append(stationen.belegt[s])
        schlange_verlauf[s].append(len(stationen.queue[s]))

tick = 0
messen()
aufnehmen(tick)
while not all(g.angekommen(tick) for g in gruppen):
    tick += 1
    for g in gruppen:
        if(random.randint(0,20) != 1):
            g.schritt(G, tick, stationen)
    aufnehmen(tick)
    messen()


# Auswertung
print("Wartezeit pro Gruppe:")
for g in gruppen:
    print(f"  {g.name}: {g.wartezeit} Ticks")
print(f"Gesamte Wartezeit: {sum(g.wartezeit for g in gruppen)} Ticks")
print(f"Alle im Ziel nach Tick {tick}")

# ----------------------------------------------------------------------
# Animation
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 7))
nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#bbbbbb", width=2)
nx.draw_networkx_nodes(G, pos, ax=ax, node_color="#f2f2f2",
                       edgecolors="#888888", node_size=1600)
nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)
nx.draw_networkx_edge_labels(G, pos, ax=ax, font_size=7,
                             edge_labels={(u, v): d["dauer"] for u, v, d in G.edges(data=True)})
ax.set_axis_off()

# Farben pro Gruppentyp (Anfangsbuchstabe des Namens)
FARBEN = {
    "S": "tab:red",
    "M": "tab:blue",
    "K": "tab:orange",
    "D": "tab:green",   # neue Gruppentypen: einfach hier ergänzen
}
STANDARDFARBE = "gray"

farben = [FARBEN.get(g.name[0], STANDARDFARBE) for g in gruppen]

punkte = ax.scatter([x for x, _, _, _ in bilder[0]],
                    [y for _, y, _, _ in bilder[0]],
                    s=140, c=farben, edgecolors="black", zorder=5)
labels = [ax.text(0, 0, g.name, fontsize=7, zorder=6, ha="center", va="bottom")
          for g in gruppen]
titel = ax.set_title("")


def update(t):
    daten = bilder[t]
    punkte.set_offsets([(x, y) for x, y, _, _ in daten])
    # noch nicht gestartete Gruppen ausblenden, wartende gold und dick umrandet
    punkte.set_facecolor([f if gest else (1, 1, 1, 0)
                          for f, (_, _, gest, _) in zip(farben, daten)])
    punkte.set_edgecolor([("gold" if wart else "black") if gest else (1, 1, 1, 0)
                          for _, _, gest, wart in daten])
    punkte.set_linewidth([3 if wart else 0.8 for _, _, _, wart in daten])
    for lab, (x, y, gest, _) in zip(labels, daten):
        lab.set_position((x, y + 0.03))
        lab.set_visible(gest)
    titel.set_text(f"Tick {t}")
    return punkte, titel, *labels



name = input("Wie soll der Outout heißen?")
print(f"Ok, Output heißt" + name + ", n=" + str(n))
SCHRITT = 2  # nur jeden 2. Tick zeigen = doppelt so schnell
frames = range(0, len(bilder), SCHRITT)
anim = FuncAnimation(fig, update, frames=frames, interval=60, blit=False)

GIF = Path("gif")
# Speichern als GIF (funktioniert ohne Zusatzprogramm):
anim.save(GIF/ (name + ", n=" + str(n) + ".gif"), writer="pillow", fps=20)
# Oder als MP4 (braucht ffmpeg):
# anim.save("rallye.mp4", writer="ffmpeg", fps=20, dpi=150)

plt.show()

# --- Auslastung pro Station ---
fig2, achsen = plt.subplots(len(stationen_liste), 1, sharex=True,
                            figsize=(12, 2.2 * len(stationen_liste)))
achsen = np.atleast_1d(achsen)

for ax_s, s in zip(achsen, stationen_liste):
    ticks = range(len(belegung_verlauf[s]))
    ax_s.step(ticks, belegung_verlauf[s], where="post", color="tab:blue", label="an der Station")
    ax_s.step(ticks, schlange_verlauf[s], where="post", color="tab:red", label="in der Warteschlange")
    ax_s.axhline(G.nodes[s]["kapazitaet"], color="gray", linestyle="--", label="Kapazität")
    ax_s.set_ylabel(s, rotation=0, ha="right", va="center")
    ax_s.set_ylim(0, max(max(schlange_verlauf[s]), G.nodes[s]["kapazitaet"]) + 1)
    ax_s.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax_s.grid(alpha=0.3)

achsen[0].legend(loc="upper right", ncol=3, fontsize=8)
achsen[-1].set_xlabel("Tick")
fig2.suptitle("Auslastung pro Station")
fig2.tight_layout()
PNG = Path("png")
fig2.savefig(PNG / (name + " auslastung, n=" + str(n) + ".png"), dpi=150)   # optional speichern