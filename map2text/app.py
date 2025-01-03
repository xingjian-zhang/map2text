import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

GRID_COLOR = "rgba(128,128,128,0.1)"
POINT_COLOR = "rgba(128,128,128,0.9)"


def rewrap_br(text, **kwargs):
    text = textwrap.fill(text, **kwargs)
    return text.replace("\n", "<br>")


def load_data(dataset_name):
    project_root = Path(__file__).parent.parent
    data_files = {
        "persona": {
            "text": "persona.tsv",
            "embedding": "persona.npz",
            "target_col": "persona",
        },
        "cs-research-idea": {
            "text": "massw.tsv",
            "embedding": "key_ideas.npz",
            "target_col": "key_idea",
        },
        "cs-research-context": {
            "text": "massw.tsv",
            "embedding": "context.npz",
            "target_col": "context",
        },
        "red_team_attempts": {
            "text": "red_team_attempts.tsv",
            "embedding": "red_team_attempts.npz",
            "target_col": "content",
        },
    }
    data_file = data_files[dataset_name]
    text_path = project_root / "data" / data_file["text"]
    embedding_path = project_root / "data" / data_file["embedding"]

    df = pd.read_csv(text_path, sep="\t")
    df = df.dropna(subset=[data_file["target_col"]])
    df.rename(columns={data_file["target_col"]: "display_text"}, inplace=True)
    df["display_text"] = df["display_text"].apply(rewrap_br, width=80)
    embeddings = np.load(embedding_path)
    return df, embeddings["low_dim_embeddings"]


def generate_plotly_figure(dataset_name):
    df, embeddings = load_data(dataset_name)
    df["x"] = embeddings[:, 0]
    df["y"] = embeddings[:, 1]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["x"],
            y=df["y"],
            mode="markers",
            marker=dict(size=2, color=POINT_COLOR),
            opacity=0.5,
            hovertemplate="%{text}<extra></extra>",
            text=df["display_text"],
        )
    )
    fig.update_layout(showlegend=False)
    fig.update_xaxes(
        visible=True,
        showgrid=True,
        showline=True,
        mirror=True,
        gridwidth=1,
        gridcolor=GRID_COLOR,
        zeroline=False,
        dtick=1,
        range=[min(embeddings[:, 0]), max(embeddings[:, 0])],
    )
    fig.update_yaxes(
        visible=True,
        showgrid=True,
        showline=True,
        mirror=True,
        gridwidth=1,
        gridcolor=GRID_COLOR,
        zeroline=False,
        dtick=1,
        range=[min(embeddings[:, 1]), max(embeddings[:, 1])],
    )
    fig.update_layout(width=1000, height=1000)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# Wide mode, dark mode
st.set_page_config(layout="wide")
st.title("Map2Text Demo")

# Select dataset


col1, col2 = st.columns([1, 1])
with col2:
    dataset_name = st.selectbox(
        "Select dataset",
        ["red_team_attempts", "persona", "cs-research-idea", "cs-research-context"],
    )
    st.write("Hello")
with col1:
    fig = generate_plotly_figure(dataset_name)
    st.plotly_chart(fig, theme=None, use_container_width=False)
