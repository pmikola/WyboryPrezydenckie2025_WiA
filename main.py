# NOTE: This Program is to visualise and estimate probability of
#  PL Presidential elections validity
# Attention: Used datasets in calculations
#  https://wybory.gov.pl/prezydent2025/pl/dane_w_arkuszach
import matplotlib
import pandas as pd, numpy as np, matplotlib.pyplot as plt
matplotlib.use("TkAgg")
path = "protokoly_po_obwodach_utf8.xlsx"
df = pd.read_excel(path, sheet_name="protokoly_po_obwodach_utf8")

eligible1 = "1 tura - Liczba wyborców uprawnionych do głosowania"
ballots1 = "1 tura - Liczba wyborców, którym wydano karty do głosowania w\xa0lokalu wyborczym oraz w\xa0głosowaniu korespondencyjnym"
eligible2 = "2 tura – Liczba wyborców uprawnionych do głosowania"
ballots2 = "2 tura – Liczba wyborców, którym wydano karty do głosowania w\xa0lokalu wyborczym oraz w\xa0głosowaniu korespondencyjnym"

candidates = {
    "Nawrocki": {
        "c1": "1 tura Nawrocki",
        "c2": "2 tura Nawrocki",
        "rozb": "Rozbieżność (2 tura – (1+Przepływ))"
    },
    "Trzaskowski": {
        "c1": "1 tura Trzaskowski",
        "c2": "2 tura Trzaskowski",
        "rozb": "Rozbieżność (2 tura – (1+Przepływ)).1"
    }
}

df["frekw_1"] = df[ballots1] / df[eligible1]
df["frekw_2"] = df[ballots2] / df[eligible2]
df["delta_turn"] = df["frekw_2"] - df["frekw_1"]

fractional = pd.concat([(df["frekw_1"] * 100) % 1, (df["frekw_2"] * 100) % 1])
plt.figure(figsize=(8, 6))
bins = np.linspace(0, 1, 21)
plt.hist(fractional, bins=bins, edgecolor="k")
plt.axhline(len(fractional) / 20, linestyle="--")
plt.xticks(np.linspace(0, 1, 11), [f"{x:.1f}" for x in np.linspace(0, 1, 11)])
plt.xlabel("Część dziesiętna frekwencji (%)")
plt.ylabel("Liczba obwodów")
plt.title("Histogram części dziesiętnej frekwencji (obie tury)")
plt.tight_layout()
plt.show()
# plt.savefig("fractions_hist.png", dpi=300)

for name, d in candidates.items():
    sigma = df[d["rozb"]].std()
    plt.figure(figsize=(8, 6))
    plt.hist(df[d["rozb"]], bins=50, edgecolor="k")
    plt.axvline(3 * sigma, linestyle="--")
    plt.axvline(-3 * sigma, linestyle="--")
    plt.xlabel(f"Rozbieżność dla {name} (głosów)")
    plt.ylabel("Liczba obwodów")
    plt.title(f"Rozbieżność modelu przepływów vs wynik – {name}")
    plt.tight_layout()
    # plt.savefig(f"rozb_{name}.png", dpi=300)
    plt.show()

    df[f"udz1_{name}"] = df[d["c1"]] / df[ballots1]
    df[f"udz2_{name}"] = df[d["c2"]] / df[ballots2]
    df[f"delta_share_{name}"] = df[f"udz2_{name}"] - df[f"udz1_{name}"]
    z = (df[f"delta_share_{name}"] - df[f"delta_share_{name}"].mean()) / df[f"delta_share_{name}"].std()
    plt.figure(figsize=(8, 6))
    plt.scatter(df["delta_turn"], df[f"delta_share_{name}"], s=6, alpha=0.3)
    plt.scatter(df.loc[abs(z) > 3, "delta_turn"], df.loc[abs(z) > 3, f"delta_share_{name}"],
                color="red", s=20, edgecolor="k")
    plt.axhline(0, linestyle=":")
    plt.axvline(0, linestyle=":")
    plt.xlabel("Zmiana frekwencji (II – I tura)")
    plt.ylabel(f"Zmiana udziału {name} (II – I tura)")
    plt.title(f"Δ udział vs Δ frekwencja – {name}")
    plt.tight_layout()
    plt.show()
    # plt.savefig(f"delta_{name}.png", dpi=300)

plik = "protokoly_po_obwodach_utf8.xlsx"
df   = pd.read_excel(plik, sheet_name="protokoly_po_obwodach_utf8")

elig1 = "1 tura - Liczba wyborców uprawnionych do głosowania"
ball1 = "1 tura - Liczba wyborców, którym wydano karty do głosowania w\xa0lokalu wyborczym oraz w\xa0głosowaniu korespondencyjnym"
elig2 = "2 tura – Liczba wyborców uprawnionych do głosowania"
ball2 = "2 tura – Liczba wyborców, którym wydano karty do głosowania w\xa0lokalu wyborczym oraz w\xa0głosowaniu korespondencyjnym"

naw1 = "1 tura Nawrocki"
naw2 = "2 tura Nawrocki"
trz1 = "1 tura Trzaskowski"
trz2 = "2 tura Trzaskowski"

df = df[(df[ball1] > 0) & (df[ball2] > 0)]

df["share_naw1"] = df[naw1] / df[ball1]
df["share_naw2"] = df[naw2] / df[ball2]
df["share_trz1"] = df[trz1] / df[ball1]
df["share_trz2"] = df[trz2] / df[ball2]

df["diff1"]       = df["share_naw1"] - df["share_trz1"]
df["diff2"]       = df["share_naw2"] - df["share_trz2"]
df["swing"]       = df["diff2"] - df["diff1"]

df["turn_1"]      = df[ball1] / df[elig1]
df["turn_2"]      = df[ball2] / df[elig2]
df["delta_turn"]  = df["turn_2"] - df["turn_1"]

plt.figure(figsize=(8, 6))
hb = plt.hexbin(df["diff1"], df["diff2"], gridsize=60,
                cmap="inferno", mincnt=1)
plt.plot([-1, 1], [-1, 1], linestyle="--", color="red", linewidth=1)
plt.xlabel("Różnica udziałów Nawrocki–Trzaskowski (I tura)")
plt.ylabel("Różnica udziałów Nawrocki–Trzaskowski (II tura)")
plt.title("Przepływ preferencji między turami\n(obwody wyborcze)")
plt.colorbar(hb, label="Liczba obwodów")
plt.tight_layout()
# plt.savefig("heat_przeplyw.png", dpi=300)
plt.show()

plt.figure(figsize=(8, 6))
hb2 = plt.hexbin(df["delta_turn"], df["swing"], gridsize=60,
                 cmap="viridis", mincnt=1)
plt.axhline(0, linestyle=":", color="white", linewidth=1)
plt.axvline(0, linestyle=":", color="white", linewidth=1)
plt.xlabel("Zmiana frekwencji (II – I)")
plt.ylabel("Swing (Δ przewagi Nawrocki–Trzaskowski)")
plt.title("‚Pochodna’ przepływu względem zmiany frekwencji")
plt.colorbar(hb2, label="Liczba obwodów")
plt.tight_layout()
plt.show()
# plt.savefig("heat_swing_vs_turnout.png", dpi=300)

x = 0.5

f = "protokoly_po_obwodach_utf8.xlsx"
df = pd.read_excel(f, sheet_name="protokoly_po_obwodach_utf8")

b1 = "1 tura - Liczba wyborców, którym wydano karty do głosowania w\xa0lokalu wyborczym oraz w\xa0głosowaniu korespondencyjnym"
b2 = "2 tura – Liczba wyborców, którym wydano karty do głosowania w\xa0lokalu wyborczym oraz w\xa0głosowaniu korespondencyjnym"
n1 = "1 tura Nawrocki"
n2 = "2 tura Nawrocki"
t1 = "1 tura Trzaskowski"
t2 = "2 tura Trzaskowski"

df = df[(df[b1] > 0) & (df[b2] > 0)]

df["ds_naw"] = (df[n2] / df[b2]) - (df[n1] / df[b1])
df["ds_trz"] = (df[t2] / df[b2]) - (df[t1] / df[b1])
mask = (abs(df["ds_naw"]) > x) | (abs(df["ds_trz"]) > x)

plt.figure(figsize=(8, 6))
plt.scatter(df.loc[~mask, b1], df.loc[~mask, b2], s=6, alpha=0.3, c="green", label=f"|Δ udział| ≤ {x*100:.0f} pp")
plt.scatter(df.loc[mask, b1], df.loc[mask, b2], s=8, alpha=0.7, c="red", label=f"|Δ udział| > {x*100:.0f} pp")
plt.xlabel("Wydane karty – I tura")
plt.ylabel("Wydane karty – II tura")
plt.title("Obwody z dużą zmianą udziału Nawrocki/Trzaskowski")
plt.legend(frameon=False)
plt.grid(True, linestyle=":")
plt.tight_layout()
plt.show()
# plt.savefig("cards_vs_cards_highlight.png", dpi=300)

x = 0.5

f = "protokoly_po_obwodach_utf8.xlsx"
df = pd.read_excel(f, sheet_name="protokoly_po_obwodach_utf8")

b1 = "1 tura - Liczba wyborców, którym wydano karty do głosowania w\xa0lokalu wyborczym oraz w\xa0głosowaniu korespondencyjnym"
b2 = "2 tura – Liczba wyborców, którym wydano karty do głosowania w\xa0lokalu wyborczym oraz w\xa0głosowaniu korespondencyjnym"
n1 = "1 tura Nawrocki"
n2 = "2 tura Nawrocki"
t1 = "1 tura Trzaskowski"
t2 = "2 tura Trzaskowski"

df = df[(df[b1] > 0) & (df[b2] > 0)]

share_n1 = df[n1] / df[b1]
share_n2 = df[n2] / df[b2]
share_t1 = df[t1] / df[b1]
share_t2 = df[t2] / df[b2]

ratio1 = share_t1 / share_n1
ratio2 = share_t2 / share_n2
delta_ratio = ratio2 - ratio1

mask = abs(delta_ratio) > x

plt.figure(figsize=(8, 6))
plt.scatter(df.loc[~mask, b1], df.loc[~mask, b2], s=6, alpha=0.3, c="green",
            label=f"|Δ (T/N)| ≤ {x:.2f}")
plt.scatter(df.loc[mask, b1], df.loc[mask, b2], s=8, alpha=0.7, c="red",
            label=f"|Δ (T/N)| > {x:.2f}")
plt.xlabel("Wydane karty – I tura")
plt.ylabel("Wydane karty – II tura")
plt.title("Obwody z dużą zmianą relacji Trzaskowski/Nawrocki")
plt.legend(frameon=False)
plt.grid(True, linestyle=":")
plt.tight_layout()
plt.show()
# plt.savefig("cards_ratio_highlight.png", dpi=300)