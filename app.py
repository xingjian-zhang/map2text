import json
import os
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
import yaml

GRID_COLOR = "rgba(128,128,128,0.1)"
POINT_COLOR = "blue"

# https://github.com/VikParuchuri/marker/issues/442#issuecomment-2636393925
torch.classes.__path__ = []

# Initialize session state variables if they don't exist.
if "random_coords" not in st.session_state:
    st.session_state.random_coords = None
if "marked_coords" not in st.session_state:
    st.session_state.marked_coords = None
if "generated_text" not in st.session_state:
    st.session_state.generated_text = None
if "selected_text" not in st.session_state:
    st.session_state.selected_text = None


def rewrap_br(text, **kwargs):
    text = textwrap.fill(text, **kwargs)
    return text.replace("\n", "<br>")


@st.cache_data
def load_data(dataset_name):
    project_root = Path(__file__).parent
    # Restrict to datasets in the 'demo' folder
    data_files = {
        "Persona": {
            "text": "demo/persona.tsv",
            "target_col": "display_text",
        },
        "Red-Teaming Strategies": {
            "text": "demo/red_teaming.tsv",
            "target_col": "display_text",
        },
    }

    data_file = data_files[dataset_name]
    text_path = project_root / data_file["text"]

    # Load the TSV file with specified columns
    df = pd.read_csv(text_path, sep="\t", usecols=["x", "y", "display_text"])
    return df


def generate_plotly_figure(df):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["x"],
            y=df["y"],
            mode="markers",
            marker=dict(size=3, color=POINT_COLOR),
            opacity=0.4,
            hovertemplate="%{customdata}<extra></extra>",
            customdata=df["display_text"],
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Rockwell",
                font_color="black",
                namelength=0
            ),
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
        range=[min(df["x"]), max(df["x"])],
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
        range=[min(df["y"]), max(df["y"])],
    )
    fig.update_layout(
        height=900,
        width=900,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(bgcolor="white", font_family="Rockwell", font_color="black"),
    )
    fig.update_layout(clickmode="event+select")
    fig.update_layout(
        hoverdistance=5,
    )
    return fig


def load_generator(generator_type, dataset_name, data):
    from map2text.model import non_trainable_gen

    project_root = Path(__file__).parent
    generation_configs = {
        "1st-order RAG": "gpt4o_zs.yaml",
        "1st-order RAG (few-shot)": "gpt4o_fs.yaml",
        "2nd-order RAG (few-shot)": "gpt4o_fs_rag.yaml",
        "1st-order RAG (CoT)": "gpt4o_fs_cot.yaml",
    }
    config_file = (
        project_root
        / "map2text"
        / "configs"
        / dataset_name.replace("-", "_")
        / generation_configs[generator_type]
    )
    with open(config_file, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # Set the target based on the dataset name.
    if dataset_name == "red_team_attempts":
        target = "red teaming strategy"
    elif dataset_name == "persona":
        target = "persona"
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    # Initialize the generator.
    generator_type = config["method"]["type"]
    if generator_type == "prompting":
        config["method"]["init_args"]["api_kwargs"]["api_key"] = os.environ[
            "OPENAI_API_KEY"
        ]
        generator = non_trainable_gen.RetrievalAugmentedGenerator(
            target=target,
            n_dims=2,
            texts=data["display_text"],
            low_dim_embeddings=data[["x", "y"]].values,
            times=np.zeros(data.shape[0]),
            **config["method"]["init_args"],
        )
    else:
        raise ValueError(f"Unknown generator type: {generator_type}.")

    return generator


st.set_page_config(layout="wide")

# Initialize marked coordinates
marked_coords = None
generated_text = None
os.environ["OPENAI_API_KEY"] = ""

col1, col2 = st.columns([1, 2])

with col1:
    st.title("MapExplorer Demo")

    st.subheader("STEP 1: Select dataset and generation method")
    subcol1, subcol2 = st.columns([1, 1])
    with subcol1:
        # Select dataset
        dataset_name = st.selectbox(
            "Select dataset",
            [
                "Red-Teaming Strategies",
                "Persona",
            ],
        )
        name_to_folder = {
            "Red-Teaming Strategies": "red_team_attempts",
            "Persona": "persona",
        }
        name_to_response = {
            "Red-Teaming Strategies": "content",
            "Persona": "persona",
        }
        df = load_data(dataset_name)

    with subcol2:
        generator_type = st.selectbox(
            "Select generation method",
            [
                "1st-order RAG",
                "1st-order RAG (few-shot)",
                "2nd-order RAG (few-shot)",
                "1st-order RAG (CoT)",
            ],
        )

    st.subheader("STEP 2: Enter OpenAI API key")
    os.environ["OPENAI_API_KEY"] = st.text_input(
        "(Required) We will not store your API key."
    )

    st.subheader("STEP 3: Generate")
    st.markdown("#### Option 1: Specify the coordinates")
    subcol1, subcol2 = st.columns([1, 1])
    with subcol1:
        x_coord = st.text_input("Enter X coordinate:")

    with subcol2:
        y_coord = st.text_input("Enter Y coordinate:")

    # Button to mark coordinate
    if st.button("Generate"):
        if os.environ["OPENAI_API_KEY"] == "":
            st.error("Please enter an OpenAI API key.")
        elif x_coord and y_coord:
            try:
                x = float(x_coord)
                y = float(y_coord)
                marked_coords = (x, y)
            except ValueError:
                st.error("Please enter valid numeric values for the coordinates.")
        else:
            st.error("Both X and Y coordinates must be provided.")

    st.markdown("#### Option 2: Random generation")
    if st.button("Generate at Random Position"):
        if os.environ["OPENAI_API_KEY"] == "":
            st.error("Please enter an OpenAI API key.")
        else:
            # Sample a point within a neighborhood of a random point
            idx = np.random.randint(0, df.shape[0])
            x_center, y_center = df.iloc[idx]["x"], df.iloc[idx]["y"]

            # Uniformly sample a random point within a radius
            radius = 0.25
            theta = np.random.uniform(0, 2 * np.pi)  # Random angle
            r = np.sqrt(np.random.uniform(0, radius))  # sqrt for uniform distribution

            x_new = x_center + r * np.cos(theta)
            y_new = y_center + r * np.sin(theta)
            st.session_state.random_coords = (x_new, y_new)  # Persist random_coords

    # Ask for confirmation if random coordinates are generated
    if st.session_state.random_coords is not None:
        st.write(
            f"Random position generated: X = {st.session_state.random_coords[0]:.2f}, Y = {st.session_state.random_coords[1]:.2f}. "
            "See the red star on the map."
        )
        st.write(
            "Click 'Confirm Random Position' to generate, or click 'Generate at Random Position' to generate a new position."
        )
        if st.button("Confirm Random Position"):
            st.session_state.marked_coords = (
                st.session_state.random_coords
            )  # Persist coordinates
            with st.spinner("Processing...(This may take about 10 seconds.)"):
                generator = load_generator(
                    generator_type,
                    name_to_folder[dataset_name],
                    df,
                )
                queries = np.array(st.session_state.marked_coords)[None, ...]
                generated_text = generator.decode_all(queries)[0][0]

                generated_text = json.loads(generated_text)
                st.session_state.generated_text = generated_text["predictions"][0][
                    name_to_response[dataset_name]
                ]

# Generate plot
fig = generate_plotly_figure(df)
if st.session_state.marked_coords:
    fig.add_trace(
        go.Scatter(
            x=[st.session_state.marked_coords[0]],
            y=[st.session_state.marked_coords[1]],
            mode="markers",
            marker=dict(size=10, color="rgb(255, 75, 75)", opacity=0.9, symbol="star"),
            hovertemplate="%{text}<extra></extra>",
            text=[rewrap_br(st.session_state.generated_text)],
            hoverlabel=dict(bgcolor="white"),
        )
    )
# Add preview point if random coordinates exist but aren't confirmed
elif st.session_state.random_coords:
    fig.add_trace(
        go.Scatter(
            x=[st.session_state.random_coords[0]],
            y=[st.session_state.random_coords[1]],
            mode="markers",
            marker=dict(size=10, color="rgb(255, 75, 75)", opacity=0.9, symbol="star"),
            hovertemplate="Preview point<br>Click 'Confirm Random Position' to generate<extra></extra>",
        )
    )

with col2:
    # Simple plot without event handling
    st.plotly_chart(
        fig,
        theme=None,
        use_container_width=False,
        use_container_height=True
    )

with col1:
    # Add the generated content section below the plot
    if st.session_state.generated_text is not None:
        st.header("Generated Content")
        st.success(st.session_state.generated_text, icon="💡")
