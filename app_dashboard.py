# ============================================================================
# Tableau de bord interactif - Marche immobilier Trulia (sept-oct 2019)
# Lancer avec : streamlit run app_dashboard.py
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="Dashboard Immobilier Trulia",
    layout="wide",
    page_icon="🏠"
)

COULEUR_PRINCIPALE = "#2E5EAA"
COULEUR_ACCENT = "#E8703A"
PALETTE = [
    COULEUR_PRINCIPALE,
    "#6FA8DC",
    "#93C47D",
    "#D5A6BD",
    "#F1C232",
    "#8E7CC3"
]

CSV_PATH = "marketing_sample_for_trulia_com-real_estate__20190901_20191031__30k_data.csv"


# ----------------------------------------------------------------------------
# Chargement + nettoyage
# ----------------------------------------------------------------------------

@st.cache_data
def charger_et_nettoyer(chemin):

    colonnes_utiles = [
        'Uniq Id',
        'Price',
        'Sqr Ft',
        'Longitude',
        'Latitude',
        'Lot Size',
        'Beds',
        'Bath',
        'Year Built',
        'Price Sqr Ft',
        'Last Sold Year',
        'Last Sold For',
        'City',
        'State',
        'Zipcode',
        'Days On Trulia'
    ]

    df = pd.read_csv(
        chemin,
        usecols=colonnes_utiles,
        low_memory=False
    )

    # Détection des locations
    df['Est_Location'] = (
        df['Price']
        .astype(str)
        .str.contains('/mo', na=False)
    )

    # Nettoyage du prix
    prix = (
        df['Price']
        .astype(str)
        .str.replace(r'[\$,+]', '', regex=True)
        .str.replace('/mo', '', regex=False)
    )

    df['Price'] = pd.to_numeric(prix, errors='coerce')

    # Nettoyage de la surface
    df['Sqr Ft'] = pd.to_numeric(
        df['Sqr Ft']
        .astype(str)
        .str.replace(',', '', regex=False)
        .str.replace('sqft', '', regex=False),
        errors='coerce'
    )

    # Bornes de nettoyage
    bornes = {
        'Price': (10_000, 10_000_000),
        'Sqr Ft': (200, 15_000),
        'Beds': (1, 10),
        'Bath': (1, 10),
        'Year Built': (1800, 2019),
    }

    for col, (lo, hi) in bornes.items():
        df.loc[
            (df[col] < lo) | (df[col] > hi),
            col
        ] = np.nan

    # Suppression des locations
    df = df[~df['Est_Location']].copy()

    # Suppression des valeurs manquantes importantes
    df = df.dropna(
        subset=[
            'Price',
            'Sqr Ft',
            'Beds',
            'Bath',
            'State',
            'City',
            'Latitude',
            'Longitude'
        ]
    )

    # Segmentation des prix
    def segmenter(p):
        if p < 150_000:
            return 'Economique'
        elif p < 300_000:
            return 'Moyen'
        elif p < 600_000:
            return 'Superieur'
        return 'Luxe'

    df['Segment Prix'] = df['Price'].apply(segmenter)

    return df.reset_index(drop=True)


# ----------------------------------------------------------------------------
# Fonction d'échantillonnage
# ----------------------------------------------------------------------------

def tirer_echantillon(df, methode, taille, seed=42):

    taille = min(taille, len(df))

    if methode == "Population complete (pas d'echantillonnage)":
        return df

    if methode == "Aleatoire simple":
        return df.sample(
            n=taille,
            random_state=seed
        )

    if methode == "Stratifie (par Etat)":

        fraction = taille / len(df)

        parts = [
            g.sample(
                n=min(
                    max(1, round(len(g) * fraction)),
                    len(g)
                ),
                random_state=seed
            )
            for _, g in df.groupby(
                'State',
                observed=True
            )
        ]

        return pd.concat(parts)

    if methode == "Systematique":

        pas_k = max(
            1,
            len(df) // taille
        )

        rng = np.random.default_rng(seed)

        depart = rng.integers(
            0,
            pas_k
        )

        return df.iloc[
            np.arange(
                depart,
                len(df),
                pas_k
            )
        ]

    if methode == "En grappes (par ville)":

        rng = np.random.default_rng(seed)

        villes = df['City'].unique().tolist()

        ordre = rng.permutation(
            len(villes)
        )

        morceaux = []
        total = 0

        for i in ordre:

            bloc = df[
                df['City'] == villes[i]
            ]

            morceaux.append(bloc)

            total += len(bloc)

            if total >= taille:
                break

        return pd.concat(morceaux)

    if methode == "Par quotas (segment de prix)":

        quotas = {
            'Economique': 0.25,
            'Moyen': 0.30,
            'Superieur': 0.30,
            'Luxe': 0.15
        }

        morceaux = []

        for seg, part in quotas.items():

            bloc = df[
                df['Segment Prix'] == seg
            ]

            q = min(
                int(taille * part),
                len(bloc)
            )

            if q > 0:
                morceaux.append(
                    bloc.sample(
                        n=q,
                        random_state=seed
                    )
                )

        return pd.concat(morceaux)

    return df


# ----------------------------------------------------------------------------
# Chargement des données
# ----------------------------------------------------------------------------

try:

    df_source = charger_et_nettoyer(
        CSV_PATH
    )

except FileNotFoundError:

    st.error(
        f"Fichier introuvable : {CSV_PATH}. "
        "Placez le CSV à côté de app_dashboard.py."
    )

    st.stop()


# ----------------------------------------------------------------------------
# BARRE LATERALE : filtres dynamiques
# ----------------------------------------------------------------------------

st.sidebar.header("🎛️ Filtres")

methode_ech = st.sidebar.selectbox(
    "Méthode d'échantillonnage",
    [
        "Population complete (pas d'echantillonnage)",
        "Aleatoire simple",
        "Stratifie (par Etat)",
        "Systematique",
        "En grappes (par ville)",
        "Par quotas (segment de prix)"
    ],
    help=(
        "Compare l'effet de chaque méthode vue en cours "
        "sur les indicateurs affichés."
    ),
)

taille_ech = st.sidebar.slider(
    "Taille de l'échantillon",
    500,
    min(10000, len(df_source)),
    3000,
    step=500
)

etats_dispo = sorted(
    df_source['State'].unique()
)

etats_choisis = st.sidebar.multiselect(
    "État(s)",
    etats_dispo,
    default=[]
)

prix_min = int(
    df_source['Price'].min()
)

prix_max = int(
    df_source['Price'].max()
)

plage_prix = st.sidebar.slider(
    "Fourchette de prix ($)",
    prix_min,
    prix_max,
    (prix_min, prix_max),
    step=5000
)

chambres_min = int(
    df_source['Beds'].min()
)

chambres_max = int(
    df_source['Beds'].max()
)

plage_chambres = st.sidebar.slider(
    "Nombre de chambres",
    chambres_min,
    chambres_max,
    (chambres_min, chambres_max)
)


# ----------------------------------------------------------------------------
# Application de l'échantillonnage puis des filtres
# ----------------------------------------------------------------------------

df_ech = tirer_echantillon(
    df_source,
    methode_ech,
    taille_ech
)

df_filtre = df_ech[
    (df_ech['Price'].between(*plage_prix))
    &
    (df_ech['Beds'].between(*plage_chambres))
]

if etats_choisis:

    df_filtre = df_filtre[
        df_filtre['State'].isin(
            etats_choisis
        )
    ]


# ----------------------------------------------------------------------------
# EN-TETE + KPIs
# ----------------------------------------------------------------------------

st.title(
    "🏠 Marché immobilier Trulia — Tableau de bord interactif"
)

st.caption(
    "Annonces collectées entre le 01/09/2019 et le 31/10/2019 "
    "— 34 états, 676 villes"
)

if df_filtre.empty:

    st.warning(
        "Aucun bien ne correspond à ces filtres. "
        "Élargissez la sélection."
    )

    st.stop()


col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Biens affichés",
    f"{len(df_filtre):,}"
)

col2.metric(
    "Prix moyen",
    f"{df_filtre['Price'].mean():,.0f} $"
)

col3.metric(
    "Prix médian",
    f"{df_filtre['Price'].median():,.0f} $"
)

col4.metric(
    "Surface moyenne",
    f"{df_filtre['Sqr Ft'].mean():,.0f} sqft"
)

st.divider()


# ----------------------------------------------------------------------------
# GRAPHIQUES
# ----------------------------------------------------------------------------

ligne1_gauche, ligne1_droite = st.columns(2)


# ----------------------------------------------------------------------------
# Histogramme des prix
# ----------------------------------------------------------------------------

with ligne1_gauche:

    fig_hist = px.histogram(
        df_filtre,
        x="Price",
        nbins=40,
        title="Distribution du prix",
        color_discrete_sequence=[
            COULEUR_PRINCIPALE
        ]
    )

    fig_hist.update_layout(
        xaxis_title="Prix ($)",
        yaxis_title="Nombre de biens",
        title_font=dict(size=16)
    )

    st.plotly_chart(
        fig_hist,
        use_container_width=True
    )


# ----------------------------------------------------------------------------
# Top 10 états
# ----------------------------------------------------------------------------

with ligne1_droite:

    prix_par_etat = (
        df_filtre
        .groupby(
            'State',
            observed=True
        )['Price']
        .mean()
        .sort_values(
            ascending=False
        )
        .head(10)
        .reset_index()
    )

    couleurs = [
        COULEUR_ACCENT if i == 0
        else COULEUR_PRINCIPALE
        for i in range(
            len(prix_par_etat)
        )
    ]

    fig_bar = px.bar(
        prix_par_etat,
        x="Price",
        y="State",
        orientation="h",
        title="Top 10 des états par prix moyen",
        color_discrete_sequence=[
            COULEUR_PRINCIPALE
        ]
    )

    fig_bar.update_traces(
        marker_color=couleurs
    )

    fig_bar.update_layout(
        yaxis=dict(
            autorange="reversed"
        ),
        xaxis_title="Prix moyen ($)",
        yaxis_title=""
    )

    st.plotly_chart(
        fig_bar,
        use_container_width=True
    )


# ----------------------------------------------------------------------------
# Scatter surface / prix
# ----------------------------------------------------------------------------

ligne2_gauche, ligne2_droite = st.columns(2)

with ligne2_gauche:

    fig_scatter = px.scatter(
        df_filtre,
        x="Sqr Ft",
        y="Price",
        color="Beds",
        title="Prix selon la surface (couleur = nb chambres)",
        color_continuous_scale="viridis",
        opacity=0.6
    )

    fig_scatter.update_layout(
        xaxis_title="Surface (sqft)",
        yaxis_title="Prix ($)"
    )

    st.plotly_chart(
        fig_scatter,
        use_container_width=True
    )


# ----------------------------------------------------------------------------
# CARTE
# ----------------------------------------------------------------------------

with ligne2_droite:

    # IMPORTANT :
    # scatter_mapbox() a été remplacé par scatter_map()
    # et mapbox_style par map_style dans les versions
    # récentes de Plotly.

    fig_carte = px.scatter_map(
        df_filtre,
        lat="Latitude",
        lon="Longitude",
        color="Price",
        size_max=10,
        zoom=2.8,
        color_continuous_scale="Bluered",
        map_style="carto-positron",
        title="Répartition géographique des biens",
        hover_data={
            "City": True,
            "Price": True
        }
    )

    fig_carte.update_layout(
        margin=dict(
            l=0,
            r=0,
            t=40,
            b=0
        )
    )

    st.plotly_chart(
        fig_carte,
        use_container_width=True
    )


# ----------------------------------------------------------------------------
# BOXPLOT
# ----------------------------------------------------------------------------

st.divider()

fig_box = px.box(
    df_filtre,
    x="Segment Prix",
    y="Price",
    color="Segment Prix",
    category_orders={
        "Segment Prix": [
            "Economique",
            "Moyen",
            "Superieur",
            "Luxe"
        ]
    },
    color_discrete_sequence=PALETTE,
    title="Dispersion du prix par segment"
)

fig_box.update_layout(
    showlegend=False,
    xaxis_title="",
    yaxis_title="Prix ($)"
)

st.plotly_chart(
    fig_box,
    use_container_width=True
)


# ----------------------------------------------------------------------------
# INFORMATIONS FINALES
# ----------------------------------------------------------------------------

st.caption(
    "Méthode d'échantillonnage active : **"
    + methode_ech
    + "** — comparez les KPIs ci-dessus en changeant "
      "de méthode pour visualiser le biais étudié "
      "dans la section 7 du notebook."
)
