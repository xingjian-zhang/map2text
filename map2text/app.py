import json
import os
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

GRID_COLOR = "rgba(128,128,128,0.1)"
POINT_COLOR = "rgba(128,128,128,0.9)"


def rewrap_br(text, **kwargs):
    text = textwrap.fill(text, **kwargs)
    return text.replace("\n", "<br>")


@st.cache_data
def load_data(dataset_name):
    project_root = Path(__file__).parent.parent

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


@st.cache_data
def generate_plotly_figure(df):
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
    fig.update_layout(width=600, height=800)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_layout(clickmode="event+select")
    return fig


def load_generator(generator_type, dataset_name, data):
    from model import non_trainable_gen

    project_root = Path(__file__).parent
    generation_configs = {
        "Few-shot": "gpt4o_fs.yaml",
        "Few-shot with CoT": "gpt4o_fs_cot.yaml",
        "Few-shot with RAG": "gpt4o_fs_rag.yaml",
        "Zero-shot": "gpt4o_zs.yaml",
    }
    config_file = (
        project_root
        / "configs"
        / dataset_name.replace("-", "_")
        / generation_configs[generator_type]
    )
    with open(config_file, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # Load complementary data
    # Set the target based on the dataset name.
    if dataset_name == "Red-Teaming Strategies":
        target = "red teaming strategy"
    elif dataset_name == "Persona":
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
            low_dim_embeddings=data[["x", "y"]],
            times=np.zeros(data.shape[0]),
            **config["method"]["init_args"],
        )
    else:
        raise ValueError(f"Unknown generator type: {generator_type}.")

    return generator


# Wide mode, dark mode
st.set_page_config(layout="wide")

# Initialize marked coordinates
marked_coords = None
generated_text = None
os.environ["OPENAI_API_KEY"] = ""

col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    st.title("Map2Text Demo")

    st.subheader("STEP 1: Select dataset and generator")
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

    with subcol2:
        generator_type = st.selectbox(
            "Select generator",
            ["Few-shot", "Few-shot with CoT", "Few-shot with RAG", "Zero-shot"],
        )

    st.subheader("STEP 2: Enter OpenAI API key here")
    os.environ["OPENAI_API_KEY"] = st.text_input("(Required) OpenAI API Key:")

    st.subheader("STEP 3: Generate")
    st.markdown("#### Option 1: Specify the coordinates")
    subcol1, subcol2 = st.columns([1, 1])
    with subcol1:
        x_coord = st.text_input("Enter X coordinate:")

    with subcol2:
        y_coord = st.text_input("Enter Y coordinate:")

    # Button to mark coordinate
    if st.button("Generate"):
        if x_coord and y_coord:
            try:
                x = float(x_coord)
                y = float(y_coord)
                marked_coords = (x, y)
            except ValueError:
                st.error("Please enter valid numeric values for the coordinates.")
        else:
            st.error("Both X and Y coordinates must be provided.")

    df = load_data(dataset_name)

    st.markdown("#### Option 2: Random generation")
    if st.button("Generate (Random)"):
        # Sample a point within a neighborhood of a random point
        idx = np.random.randint(0, df.shape[0])
        x_center, y_center = df.iloc[idx]["x"], df.iloc[idx]["y"]

        # Uniformly sample a random point within a radius of 1
        radius = 1
        theta = np.random.uniform(0, 2 * np.pi)  # Random angle
        r = np.sqrt(
            np.random.uniform(0, radius)
        )  # Random radius (sqrt for uniform distribution)

        x_new = x_center + r * np.cos(theta)
        y_new = y_center + r * np.sin(theta)
        marked_coords = (x_new, y_new)

    # Create the Streamlit Plotly chart with optional marked point
    if marked_coords is not None:
        with st.spinner("Processing..."):
            generator = load_generator(
                generator_type,
                name_to_folder[dataset_name],
                df,
            )
            queries = np.array(marked_coords)[None, ...]
            generated_text = generator.decode_all(queries)[0][0]

            generated_text = json.loads(generated_text)
            generated_text = generated_text["predictions"][0][
                name_to_response[dataset_name]
            ]

# Generate plot
fig = generate_plotly_figure(df)
if marked_coords:
    fig.add_trace(
        go.Scatter(
            x=[marked_coords[0]],
            y=[marked_coords[1]],
            mode="markers",
            marker=dict(size=10, color="red"),
            hovertemplate="%{text}<extra></extra>",
            text=[rewrap_br(generated_text)],
        )
    )

with col2:
    st.plotly_chart(fig, theme=None, use_container_width=True)

with col3:
    st.header("Generation:")
    if generated_text is not None:
        st.markdown(
            f"""<p style="color:red; font-family: Courier New;">{generated_text}</p>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown("Please follow the instructions to specify a point.")
