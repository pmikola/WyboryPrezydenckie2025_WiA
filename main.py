# NOTE: This Program is to visualise and estimate probability of
#  PL Presidential elections validity
# Attention: Used datasets in calculations
#  https://wybory.gov.pl/prezydent2025/pl/dane_w_arkuszach
import matplotlib
import pandas as pd, numpy as np, matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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

file = "protokoly_po_obwodach_utf8.xlsx"
df   = pd.read_excel(file, sheet_name="protokoly_po_obwodach_utf8")

elig1 = "1 tura - Liczba wyborców uprawnionych do głosowania"
elig2 = "2 tura – Liczba wyborców uprawnionych do głosowania"
n2    = "2 tura Nawrocki"
t2    = "2 tura Trzaskowski"

df = df[(df[elig1] > 0) & (df[elig2] > 0)]

delta_elig = df[elig2] - df[elig1]              # przyrost uprawnionych
threshold  = np.nanpercentile(delta_elig, 99)   # górny 1 % wartości

vote_diff  = df[n2] - df[t2]
lim_vote   = np.nanpercentile(abs(vote_diff), 99)
norm_cd    = mcolors.TwoSlopeNorm(vmin=-lim_vote, vcenter=0, vmax=lim_vote)
cmap_cd    = plt.get_cmap("bwr")

# maski
top_growth = delta_elig >= threshold

plt.figure(figsize=(10, 7))

plt.scatter(df.loc[~top_growth, elig1],
            df.loc[~top_growth, elig2],
            s=6, alpha=0.2, color="grey", label="pozostałe komisje")

sc = plt.scatter(df.loc[top_growth, elig1],
                 df.loc[top_growth, elig2],
                 c=vote_diff[top_growth],
                 cmap=cmap_cd, norm=norm_cd,
                 edgecolors="black", linewidths=0.3,
                 s=30, alpha=0.9,
                 label="górny 1 % Δ uprawnionych")

max_axis = max(df[elig1].max(), df[elig2].max())
plt.plot([0, max_axis], [0, max_axis], linestyle="--", color="black", linewidth=0.7)

plt.colorbar(sc, label="Przewaga głosów (Nawrocki − Trzaskowski)  •  II tura")
plt.xlabel("Uprawnieni do głosowania – I tura")
plt.ylabel("Uprawnieni do głosowania – II tura")
plt.title("Komisje z największym przyrostem uprawnionych między turami\n"
          "(kolor: kto zyskał więcej głosów w II turze)")
plt.legend(frameon=False)
plt.tight_layout()
plt.show()



import pandas as pd, numpy as np, matplotlib.pyplot as plt, matplotlib.colors as mcolors
file = "protokoly_po_obwodach_utf8.xlsx"
df   = pd.read_excel(file, sheet_name="protokoly_po_obwodach_utf8")

b1 = "1 tura - Liczba wyborców, którym wydano karty do głosowania w\xa0lokalu wyborczym oraz w\xa0głosowaniu korespondencyjnym"
b2 = "2 tura – Liczba wyborców, którym wydano karty do głosowania w\xa0lokalu wyborczym oraz w\xa0głosowaniu korespondencyjnym"
n1, n2 = "1 tura Nawrocki", "2 tura Nawrocki"
t1, t2 = "1 tura Trzaskowski", "2 tura Trzaskowski"

df = df[(df[b1] > 0) & (df[b2] > 0)]

shr_n1, shr_n2 = df[n1]/df[b1], df[n2]/df[b2]
shr_t1, shr_t2 = df[t1]/df[b1], df[t2]/df[b2]
log_ratio      = np.log2((shr_t2/shr_n2) / (shr_t1/shr_n1))

lim   = np.nanpercentile(abs(log_ratio), 99)
norm  = mcolors.TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
cmap  = plt.get_cmap("bwr")

x = df[b2]
y = df[b2] - (df[n2] + df[t2])

flip = np.sign(df[n1]-df[t1]) != np.sign(df[n2]-df[t2])

plt.figure(figsize=(10,6))

sc = plt.scatter(x, y, c=log_ratio, cmap=cmap, norm=norm,
                 s=14, alpha=0.8, edgecolors="none")

# flip_sc = plt.scatter(x[flip], y[flip],
#                       facecolors="none",
#                       edgecolors="black",
#                       linewidths=0.1,        # cieniutka ramka
#                       alpha=0.1,             # pół-przezroczysta
#                       s=40,
#                       label="zamiana lidera (Trz ↔ Naw)")

plt.colorbar(sc, label="log₂ ( ratio Trz/Naw II ÷ ratio I )")
plt.axhline(0, linestyle=":", color="grey")
plt.xlabel("wydane karty (II tura)")
plt.ylabel("karty − głosy (II tura)")
plt.title("Karty wydane vs. zagospodarowane głosy\nKolor → zmiana relacji Trz/Naw")
# plt.legend(handles=[flip_sc], frameon=False)
plt.tight_layout()
plt.show()

for name, d in candidates.items():
    sigma = df[d["rozb"]].std()
    plt.figure(figsize=(8, 6))
    plt.hist(df[d["rozb"]], bins=100, edgecolor="k")
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
    mask = abs(z) > 3

    plt.figure(figsize=(8, 6))
    plt.scatter(df.loc[~mask, "delta_turn"],
                df.loc[~mask, f"delta_share_{name}"],
                s=6, alpha=0.3,
                label=r"$|\Delta|\;\leq\;3\sigma$")

    plt.scatter(df.loc[mask, "delta_turn"],
                df.loc[mask, f"delta_share_{name}"],
                color="red", edgecolor="k", s=20,
                label=r"$|\Delta|\;>\;3\sigma$")

    plt.axhline(0, linestyle=":")
    plt.axvline(0, linestyle=":")
    plt.xlabel("Zmiana frekwencji (II – I tura)")
    plt.ylabel(f"Zmiana udziału {name} (II – I tura)")
    plt.title(f"Δ udział vs Δ frekwencja – {name}")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()

    # plt.savefig(f"delta_{name}.png", dpi=300)

z_n   = (df["delta_share_Nawrocki"]     - df["delta_share_Nawrocki"].mean())     / df["delta_share_Nawrocki"].std()
z_t   = (df["delta_share_Trzaskowski"]  - df["delta_share_Trzaskowski"].mean())  / df["delta_share_Trzaskowski"].std()

norm_n = abs(z_n) <= 3
anom_n = abs(z_n) >  3
norm_t = abs(z_t) <= 3
anom_t = abs(z_t) >  3

plt.figure(figsize=(8, 6))

plt.scatter(df.loc[norm_n, "delta_turn"],
            df.loc[norm_n, "delta_share_Nawrocki"],
            s=6, alpha=0.3, color="tab:blue",
            label="Nawrocki |Δ| ≤ 3σ")

plt.scatter(df.loc[anom_n, "delta_turn"],
            df.loc[anom_n, "delta_share_Nawrocki"],
            marker="o", edgecolor="k", color="tab:blue", s=20,
            label="Nawrocki |Δ| > 3σ")

plt.scatter(df.loc[norm_t, "delta_turn"],
            df.loc[norm_t, "delta_share_Trzaskowski"],
            s=6, alpha=0.3, color="tab:orange",
            label="Trzaskowski |Δ| ≤ 3σ")

plt.scatter(df.loc[anom_t, "delta_turn"],
            df.loc[anom_t, "delta_share_Trzaskowski"],
            marker="^", edgecolor="k", color="tab:orange", s=20,
            label="Trzaskowski |Δ| > 3σ")

plt.axhline(0, linestyle=":")
plt.axvline(0, linestyle=":")
plt.xlabel("Zmiana frekwencji (II – I tura)")
plt.ylabel("Zmiana udziału (II – I tura)")
plt.title("Δ udział vs Δ frekwencja – Nawrocki & Trzaskowski")
plt.legend(frameon=False)
plt.tight_layout()
plt.show()

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