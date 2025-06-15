# NOTE: This Program is to visualise and estimate probability of
#  PL Presidential elections validity
# Attention: Used datasets in calculations
#  https://wybory.gov.pl/prezydent2025/pl/dane_w_arkuszach
import io
import re
import sys
import zipfile

import contextily
import geojson
import matplotlib
import pandas as pd, numpy as np, matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.graph_objects as go
import plotly.offline as pyo
import plotly.express as px
import webbrowser, os
import colorsys
from geopy import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os, math, json, time, requests, pandas as pd

from scipy.stats import gaussian_kde
from sklearn import preprocessing
from tqdm.auto import tqdm
from xml.etree import ElementTree as ET

matplotlib.use("TkAgg")
path = "protokoly_po_obwodach_utf8.xlsx"
df = pd.read_excel(path, sheet_name="protokoly_po_obwodach_utf8")
FLOW_SHEET  = "przepływy"
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

file_path = "protokoly_po_obwodach_utf8.xlsx"
sheet     = "protokoly_po_obwodach_utf8"
file_path_1 = "protokoly_po_obwodach_w_pierwszej_turze_utf8.xlsx"
sheet_1     = "Arkusz"
file_path_2 = "protokoly_po_obwodach_w_drugiej_turze_utf8.xlsx"
sheet_2     = "Arkusz"
n1 = "1 tura Nawrocki"
n2 = "2 tura Nawrocki"
t1 = "1 tura Trzaskowski"
t2 = "2 tura Trzaskowski"
b1 = "1 tura - Liczba wyborców, którym wydano karty do głosowania w\xa0lokalu wyborczym oraz w\xa0głosowaniu korespondencyjnym"
b2 = "2 tura – Liczba wyborców, którym wydano karty do głosowania w\xa0lokalu wyborczym oraz w\xa0głosowaniu korespondencyjnym"

transfer_pct_to_n = {
    "BARTOSZEWICZ": 0.673,
    "BIEJAT"      : 0.098,
    "BRAUN"       : 0.925,
    "HOŁOWNIA"    : 0.138,
    "JAKUBIAK"    : 0.903,
    "MACIAK"      : 0.729,
    "MENTZEN"     : 0.881,
    "NAWROCKI"    : 1.,
    "SENYSZYN"    : 0.189,
    "STANOWSKI"   : 0.512,
    "TRZASKOWSKI" : 0.,
    "WOCH"        : 0.654,
    "ZANDBERG"    : 0.162,
}
df_1 = pd.read_excel(file_path_1, sheet_name=sheet_1)
df_2 = pd.read_excel(file_path_2, sheet_name=sheet_2)

full_names_1 = df_1.columns[-13:].tolist()
tcfu1 = df_1.columns[-22]
total_cards_from_urns_1 = df_1[tcfu1].to_numpy(dtype=float)
tcefu1 = df_1.columns[-19]
total_cards_eligible_from_urns_1 = df_1[tcefu1].to_numpy(dtype=float)
avm1 = df_1.columns[-18]
all_votes_missed_1 = df_1[avm1].fillna(0).to_numpy(dtype=float)
cwsx1 = df_1.columns[-17]
cards_with_second_x_1 = df_1[cwsx1].fillna(0).to_numpy(dtype=float)

full_names_2 = df_2.columns[-2:].tolist()
tcfu2 = df_2.columns[-10]
total_cards_from_urns_2 = df_2[tcfu2].fillna(0).to_numpy(dtype=float)
tcefu2 = df_2.columns[-7]
total_cards_eligible_from_urns_2 = df_2[tcefu2].fillna(0).to_numpy(dtype=float)
avm2 = df_2.columns[-6]
all_votes_missed_2 = df_2[avm2].fillna(0).to_numpy(dtype=float)
cwsx2 = df_2.columns[-5]
cards_with_second_x_2 = df_2[cwsx2].fillna(0).to_numpy(dtype=float)
all_mised_2 = total_cards_from_urns_2 -total_cards_eligible_from_urns_2 + all_votes_missed_2
da2 = df_2.columns[6]
district_adresses_2 = df_2[da2]#.str.split(",", n=1, expand=True)[1].str.replace(".","")
# print(district_adresses_2[0:1])
# sys.exit()
names_1 = [fn.split()[0] for fn in full_names_1]
names_2 = [fn.split()[0] for fn in full_names_2]
first_round_votes = df_1[full_names_1].to_numpy(dtype=float)
second_round_votes = df_2[full_names_2].to_numpy(dtype=float)
avg_votes_per_cand = first_round_votes.mean(axis=0)
first_round_votes = (df_1[full_names_1].fillna(0).to_numpy(dtype=float))
total_first_votes = np.sum(first_round_votes,axis=0)


PARQUET_PATH = "pooling_districts_PL.parquet"
download_coord_data = True
TOKEN = "pk.eyJ1IjoicG1pa29sMSIsImEiOiJjbWJtaHdraWIxMmxoMmtxdjV2ZGM3a285In0.P0S7qlTM4yZq9noJaAcwzw"
tomtom_api_key = "FDdJ6DFVf9tsNx2XzddR5T12ZkmQ81W9"
google_api_key = "AIzaSyA367-N-J9P9Ash_Nm0kChdtVlnSaxPmxY"

if download_coord_data and not os.path.exists(PARQUET_PATH):
    BATCH_URL = "https://api.mapbox.com/search/geocode/v6/batch"

    queries = district_adresses_2
    CHUNK   = 10
    n_calls = math.ceil(len(queries) / CHUNK)

    def make_body(slice_):
        return [{"types": ["address"], "q": addr, "limit": 1} for addr in slice_]

    lat, lon = [], []
    for i in tqdm(range(n_calls), desc="Geocoding"):
        chunk = queries[i*CHUNK:(i+1)*CHUNK]
        body  = make_body(chunk)
        params = {"access_token": TOKEN}
        r = requests.post(BATCH_URL, params=params,
                          headers={"Content-Type": "application/json"},
                          data=json.dumps(body))
        r.raise_for_status()
        for res in r.json()["batch"]:
            if res.get("features") is not None:
                if res["features"]:
                    g = res["features"][0]["geometry"]["coordinates"]
                    lon.append(g[0]);  lat.append(g[1])
                else:                                  # no hit
                    lon.append(None); lat.append(None)
                    print(res)
                    sys.exit()
            else:  # no hit
                lon.append(None);lat.append(None)
                print(res)
                sys.exit()
        time.sleep(1)

    out_paraquet = pd.DataFrame({"query": queries,
                        "latitude": lat,
                        "longitude": lon})

    out_paraquet.to_parquet(PARQUET_PATH)
    print(f"Finished – coordinates saved to {PARQUET_PATH}")
    print(out_paraquet)
else:
    out_paraquet = pd.read_parquet(PARQUET_PATH, engine="pyarrow")#.dropna()
    # print(out_paraquet['query'])
    # sys.exit()
    filled_paraqued_path = "out_paraquet_completed.parquet"
    mask_missing_street = (
    out_paraquet['latitude'].isna()
    | out_paraquet['longitude'].isna()
    )
    adr_list = out_paraquet["query"].tolist()
    # TODO : filling missed geocords
    time.sleep(1)
    # print(mask_missing_street)


import json, unicodedata, pandas as pd, geopandas as gpd, matplotlib.pyplot as plt, matplotlib.colors as mcolors
from shapely.geometry import Point

def norm_str(t):
    t = unicodedata.normalize("NFKD", str(t) or "").casefold()
    t = " ".join(t.replace(",", " ,").split())
    return t.rstrip(",")

xls = "protokoly_po_obwodach_w_drugiej_turze_utf8.xlsx"
df_votes = pd.read_excel(xls, sheet_name="Arkusz")

with open("exp_obwody.geojson", encoding="utf-8") as f:
    gj = json.load(f)

coord_lookup = {norm_str(ft["properties"].get("pelna_siedziba", "")): ft["geometry"]["coordinates"]
                for ft in gj["features"] if ft.get("geometry")}

df_votes[["x2180", "y2180"]] = df_votes["Siedziba"].apply(
    lambda a: pd.Series(coord_lookup.get(norm_str(a), (pd.NA, pd.NA))))

naw = "NAWROCKI Karol Tadeusz"
trz = "TRZASKOWSKI Rafał Kazimierz"

df_votes["score"] = ((all_votes_missed_2 - all_votes_missed_1) )#(df_votes[trz] - df_votes[naw]) / (df_votes[trz] + df_votes[naw])
df_votes["tot"]   = all_votes_missed_2#df_votes[trz] + df_votes[naw]

tv_min, tv_max = df_votes["tot"].min(), df_votes["tot"].max()
df_votes["msize"] = 4 + 16 * (df_votes["tot"] - tv_min) / (tv_max - tv_min)


df_ok = df_votes.dropna(subset=["x2180", "y2180", "score"])
gdf_pts = gpd.GeoDataFrame(
    df_ok,
    geometry=[Point(x, y) for x, y in zip(df_ok.x2180, df_ok.y2180)],
    crs="EPSG:2180"
).to_crs(4326)
vmin, vmax = gdf_pts["score"].min(), gdf_pts["score"].max()

gdf_muni = gpd.read_file( "PRG_jednostki_administracyjne_2024/PRG_jednostki_administracyjne_2024/A06_Granice_obrebow_ewidencyjnych.shp").to_crs(4326)
# gdf_muni = gpd.read_file( "PRG_jednostki_administracyjne_2024/PRG_jednostki_administracyjne_2024/A03_Granice_gmin.shp").to_crs(4326)

cmap  = mcolors.LinearSegmentedColormap.from_list("bo", ["green", "red"])
normc = plt.Normalize(vmin=vmin, vmax=vmax)

fig, ax = plt.subplots(figsize=(8, 8))
gdf_muni.plot(ax=ax, edgecolor="grey", facecolor="none", linewidth=0.1)

sc = ax.scatter(
    gdf_pts.geometry.x, gdf_pts.geometry.y,
    c=gdf_pts["score"], cmap=cmap, norm=normc,marker='.',
    s=gdf_pts["msize"] * 0.8, alpha=0.8, edgecolors="none"
)

cbar = fig.colorbar(sc, ax=ax, label="Stosunek głosów nieważnych\n I vs. II tura")
ax.set_axis_off()
# plt.style.use('dark_background')
ax.set_title("Zmiana liczby utraconych głosów między \n I a II turą (po obwodach) (j.ewidencyjne)")
fig.text(0.5, 0.1, "Źródło: GUGIK na podstawie danych PKW", ha="center", va="bottom", fontsize=8)
fig.text(0.75, 0.5, "@rezolucjonista", ha="right", va="center",rotation=30, fontsize=48, color="grey", alpha=0.25)
# plt.tight_layout()
# joined = gpd.sjoin(
#     gdf_pts[["score", "geometry"]],
#     gdf_muni[["geometry"]],
#     how="inner",
#     predicate="within"
# )
#
# mean_score = joined.groupby("index_right")["score"].mean()
# gdf_muni["mean_score"] = mean_score          # NaN tam, gdzie brak punktów
#
# # ─── 2  mapa ────────────────────────────────────────────────────────────
# cmap  = mcolors.LinearSegmentedColormap.from_list("bo", ["blue", "orange"])
# normc = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
#
# fig, ax = plt.subplots(figsize=(8, 8))
#
# gdf_muni.plot(
#     ax=ax,
#     column="mean_score",
#     cmap=cmap,
#     norm=normc,
#     edgecolor="grey",
#     linewidth=0.1,
#     missing_kwds=dict(color="white", alpha=0)   # jednostki bez danych = puste
# )
#
# fig.colorbar(
#     plt.cm.ScalarMappable(norm=normc, cmap=cmap),
#     ax=ax, label="Średni (Trz − Naw) / (Trz + Naw)"
# )
#
# ax.set_axis_off()
# ax.set_title("Średni stosunek głosów Trzaskowski / Nawrocki\nw jednostkach ewidencyjnych")
# fig.text(0.5, 0.08, "Źródło: GUGIK na podstawie danych PKW", ha="center", va="bottom", fontsize=8)
# fig.text(0.75, 0.5, "@rezolucjonista", ha="right", va="center", rotation=30,
#          fontsize=48, color="grey", alpha=0.25)

plt.savefig("trz_naw_utracone.png", format='png',dpi=1200)
plt.show()





sys.exit()
idx_naw, idx_trz = names_2.index(names_2[0]), names_2.index(names_2[1])
flows = np.zeros((*first_round_votes.shape, 2))
for k, name in enumerate(names_1):
    p = transfer_pct_to_n[name]
    flows[:, k, idx_naw]   = first_round_votes[:, k] * p
    flows[:, k, idx_trz] = first_round_votes[:, k] * (1 - p)

flow_sum = flows.sum(axis=1)
eps = 1e-6
r1 = flow_sum[:, idx_trz] / (flow_sum[:, idx_naw] + eps)
r2 = second_round_votes[:, idx_trz] / (second_round_votes[:, idx_naw] + eps)
mask = np.isfinite(r1) & np.isfinite(r2)
r1, r2 = r1[mask], r2[mask]
p1 = r1 / (1 + r1)
p2 = r2 / (1 + r2)
pct_shift = 100 * (p2 - p1)

lim = np.nanmax(np.abs(pct_shift))
norm = mcolors.TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
cmap = plt.get_cmap("coolwarm")
fig, ax = plt.subplots(figsize=(6,6))
sc = ax.scatter(p1, p2, c=pct_shift, cmap=cmap, norm=norm, s=8, alpha=0.6)
ax.plot([0,1], [0,1], "--k", lw=1)
ax.set_xlabel("Udział Trz/Naw (I tura)")
ax.set_ylabel("Udział Trz/Naw (II tura)")
ax.set_title("Zmiana udziału Trz/Naw\nz udziałem głosów innych kandydatów (z I tury)")
fig.colorbar(sc, ax=ax, label="% zmiany udziału Trz/Naw (II vs I)")
ax.grid(True, linestyle=":")

fig.text(0.45, 0.5, "@rezolucjonista", fontsize=40,
         color="black", alpha=0.05, ha="center", va="center", rotation=30)

plt.tight_layout()
plt.show()
sys.exit()
flows = np.zeros((13, 2), dtype=float)
for i, lname in enumerate(names_1):
    p = transfer_pct_to_n[lname]
    flows[i, idx_naw]   = total_first_votes[i] * p
    flows[i, idx_other] = total_first_votes[i] * (1 - p)

print(flows)
sys.exit()

src, tgt, vals = [], [], []
for i in range(13):
    for j in range(2):
        src.append(i)
        tgt.append(13 + j)
        vals.append(flows[i, j])

base_colors = px.colors.qualitative.Dark24[:13]

def adjust_lightness(hex_color, factor):
    h = hex_color.lstrip('#')
    r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
    h_l, l, s = colorsys.rgb_to_hls(r/255, g/255, b/255)
    l = max(0, min(1, l * factor))
    r2, g2, b2 = colorsys.hls_to_rgb(h_l, l, s)
    return f'#{int(r2*255):02x}{int(g2*255):02x}{int(b2*255):02x}'

link_colors = []
for i, base in enumerate(base_colors):
    # lighten for Nawrocki, darken for the other
    light = adjust_lightness(base, 1.3)
    dark  = adjust_lightness(base, 0.7)
    link_colors.extend([light, dark])

node_labels = names_1 + names_2
fig = go.Figure(go.Sankey(
    arrangement="snap",
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=node_labels
    ),
    link=dict(
        source=src,
        target=tgt,
        value=vals,
        color=link_colors
    )
))
fig.update_layout(
    title_text="Przepływ I tura → II tura (Nawrocki vs Trzaskowski)",
    font_size=12,
    width=900,
    height=600
)

out = "sankey_colored.html"
fig.write_html(out, include_plotlyjs="include", full_html=True)
webbrowser.open("file://" + os.path.abspath(out))

sys.exit()
print(names_2)


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
plt.savefig("fractions_hist.png", dpi=300)

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
b1 = "1 tura - Liczba wyborców, którym wydano karty do głosowania w\xa0lokalu wyborczym oraz w\xa0głosowaniu korespondencyjnym"
b2 = "2 tura – Liczba wyborców, którym wydano karty do głosowania w\xa0lokalu wyborczym oraz w\xa0głosowaniu korespondencyjnym"
n1, n2 = "1 tura Nawrocki", "2 tura Nawrocki"
t1, t2 = "1 tura Trzaskowski", "2 tura Trzaskowski"
df["turn_1"]     = df[b1] / df[elig1]
df["turn_2"]     = df[b2] / df[elig2]
df["delta_turn"] = df["turn_2"] - df["turn_1"]
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