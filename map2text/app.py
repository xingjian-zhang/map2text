import textwrap
from pathlib import Path

import json
import numpy as np
import os
import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

import streamlit as st
import yaml
import gdown


GRID_COLOR = "rgba(128,128,128,0.1)"
POINT_COLOR = "rgba(128,128,128,0.9)"


def rewrap_br(text, **kwargs):
    text = textwrap.fill(text, **kwargs)
    return text.replace("\n", "<br>")


@st.cache_data
def load_data(dataset_name):
    project_root = Path(__file__).parent.parent

    data_files = {
        "Persona": {
            "text": "persona.tsv",
            "embedding": "persona.npz",
            "target_col": "persona",
        },
        "CS Research Idea": {
            "text": "massw.tsv",
            "embedding": "key_ideas.npz",
            "target_col": "key_idea",
        },
        "CS Research Context": {
            "text": "massw.tsv",
            "embedding": "context.npz",
            "target_col": "context",
        },
        "Red-Teaming Strategies": {
            "text": "red_team_attempts.tsv",
            "embedding": "red_team_attempts.npz",
            "target_col": "content",
        },
    }

    file_ids = {
        "Persona": {
            "tsv": "1MjId3yn16h5jedhPyFZE-fRSo0jeikrY",
            "embedding": "1q9zFuluhSpT_ljUypCEx8XeC81zWqhTS",
        },
        "CS Research Idea": {
            "text": "1XCIR7w5JP0T49vnfSgvq2tHXjKY_amLl",
            "embedding": "1Ink96aAq44I-Xj4nDOfGC_h-8ZxuDB8q",
        },
        "CS Research Context": {
            "text": "1XCIR7w5JP0T49vnfSgvq2tHXjKY_amLl",
            "embedding": "1MuP3HjYqg2dv1gyyjHLTl0UWfCMHQN67",
        },
        "Red-Teaming Strategies": {
            "text": "1fHTH37YqgBDE1F6SZX7usDkGZG1ZiTdj",
            "embedding": "1zo-xj9xkXeO4FnIgvW1LbuUOTQSKkwLu",
        },
    }

    for f_type in ["text", "embedding"]:
        data_dir = "../data"
        os.makedirs(data_dir, exist_ok=True)
        gdown.download(
            f'https://drive.google.com/uc?export=download&id={file_ids[dataset_name][f_type]}',
            output=os.path.join(data_dir, data_files[dataset_name][f_type]), quiet=False
        )

    data_file = data_files[dataset_name]
    text_path = project_root / "data" / data_file["text"]
    embedding_path = project_root / "data" / data_file["embedding"]

    df = pd.read_csv(text_path, sep="\t")
    df = df.dropna(subset=[data_file["target_col"]])
    df.rename(columns={data_file["target_col"]: "display_text"}, inplace=True)
    df["display_text"] = df["display_text"].apply(rewrap_br, width=80)
    embeddings = np.load(embedding_path)
    return df, embeddings["low_dim_embeddings"], embeddings["high_dim_embeddings"]


@st.cache_data
def generate_plotly_figure(df, embeddings):
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
    fig.update_layout(width=600, height=800)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_layout(clickmode="event+select")
    return fig


def load_generator(generator_type, dataset_name, data, low_dim_embeddings, high_dim_embeddings):
    from model import non_trainable_gen, trainable_ffn
    project_root = Path(__file__).parent
    generation_configs = {
        "plagiarism": "plagiarism.yaml",
        "embedding": "vec2text_uw.yaml",
        "embedding_ffn": "vec2text_ffn_inference.yaml",
        "Few-shot": "gpt4o_fs.yaml",
        "Few-shot with CoT": "gpt4o_fs_cot.yaml",
        "Few-shot with RAG": "gpt4o_fs_rag.yaml",
        "Zero-shot": "gpt4o_zs.yaml",
    }
    config_file = (project_root / "configs" / dataset_name.replace("-", "_") /
                   generation_configs[generator_type])
    with open(config_file, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # Load complementary data
    times = data[config["data"]["time_col"]]
    time_split = config["data"]["time_split"]
    targets = data["display_text"]
    targets_old = targets[times < time_split].tolist()
    low_dim_embeddings_old = low_dim_embeddings[times < time_split]
    n_dims = low_dim_embeddings.shape[1]

    # Initialize the generator.
    generator_type = config["method"]["type"]
    if generator_type == "plagiarism":
        generator = non_trainable_gen.PlagiarismGenerator(
            n_dims=n_dims,
            data_old=targets_old,
            low_dim_embeddings_old=low_dim_embeddings_old,
            **config["method"]["init_args"],
        )
    elif generator_type == "embedding":
        high_dim_embeddings_old = high_dim_embeddings[times < time_split]
        generator = non_trainable_gen.EmbeddingInversionGenerator(
            n_dims=n_dims,
            data_old=targets_old,
            low_dim_embeddings_old=low_dim_embeddings_old,
            high_dim_embeddings_old=high_dim_embeddings_old,
            **config["method"]["init_args"],
        )
    elif generator_type == "embedding_ffn":
        generator = trainable_ffn.EmbeddingInversionFFNGenerator(
            n_dims=n_dims,
            data_old=targets_old,
            low_dim_embeddings_old=low_dim_embeddings_old,
            **config["method"]["init_args"],
        )
    elif generator_type == "prompting":
        # Set the api-key here
        config["method"]["init_args"]["api_kwargs"]["api_key"] = os.environ["OPENAI_API_KEY"]
        generator = non_trainable_gen.RetrievalAugmentedGenerator(
            target= config["data"]["target_col"],
            n_dims=n_dims,
            texts=targets_old,
            low_dim_embeddings=low_dim_embeddings_old,
            times=times[times < time_split].values,
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
            ["Red-Teaming Strategies", "Persona", "CS Research Idea", "CS Research Context"],
        )
        name_to_folder = {
            "Red-Teaming Strategies": "red_team_attempts",
            "Persona": "persona",
            "CS Research Idea": "cs_research_idea",
            "CS Research Context": "cs_research_context",
        }
        name_to_response = {
            "Red-Teaming Strategies": "content",
            "Persona": "persona",
            "CS Research Idea": "key_idea",
            "CS Research Context": "context",
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

    df, low_dim_embeddings, high_dim_embeddings = load_data(dataset_name)

    st.markdown("#### Option 2: Random generation")
    if st.button("Generate (Random)"):
        # Sample a point within a neighborhood of a random point
        idx = np.random.randint(0, low_dim_embeddings.shape[0])
        x_center, y_center = low_dim_embeddings[idx]

        # Uniformly sample a random point within a radius of 1
        radius = 1
        theta = np.random.uniform(0, 2 * np.pi)  # Random angle
        r = np.sqrt(np.random.uniform(0, radius))  # Random radius (sqrt for uniform distribution)

        x_new = x_center + r * np.cos(theta)
        y_new = y_center + r * np.sin(theta)
        marked_coords = (x_new, y_new)

    # Create the Streamlit Plotly chart with optional marked point
    if marked_coords is not None:
        with st.spinner("Processing..."):
            generator = load_generator(
                generator_type, name_to_folder[dataset_name], df, low_dim_embeddings, high_dim_embeddings,
            )
            queries = np.array(marked_coords)[None, ...]
            generated_text = generator.decode_all(queries)[0][0]

            generated_text = json.loads(generated_text)
            generated_text = generated_text["predictions"][0][name_to_response[dataset_name]]

# Generate plot
fig = generate_plotly_figure(df, low_dim_embeddings)
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
    st.plotly_chart(fig,
                    theme=None,
                    use_container_width=True)

with col3:
    st.header("Generation:")
    if generated_text is not None:
        st.markdown(f"""<p style="color:red; font-family: Courier New;">{generated_text}</p>""", unsafe_allow_html=True)
    else:
        st.markdown("Please follow the instructions to specify a point.")
