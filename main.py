import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
from langchain.llms import LlamaCpp
import requests

# File formats supported
file_formats = {
    "csv": pd.read_csv,
    "xls": pd.read_excel,
    "xlsx": pd.read_excel,
    "xlsm": pd.read_excel,
    "xlsb": pd.read_excel,
}


@st.cache_data(ttl=3600)
def load_data(uploaded_file):
    try:
        ext = os.path.splitext(uploaded_file.name)[1][1:].lower()
    except:
        ext = uploaded_file.split(".")[-1]
    if ext in file_formats:
        return file_formats[ext](uploaded_file)
    else:
        st.error(f"Unsupported file format: {ext}")
        return None


st.set_page_config(page_title="DataFrame Analysis")

st.title("DataFrame Analysis")

st.sidebar.write("""
How to use:
1. Load CSV or Excel file.
2. Pick your sampling method.
3. Ask LLM about the data.
4. Generate plots.
""")

uploaded_file = st.file_uploader("Load CSV or Excel file", type=["csv", "xls", "xlsx", "xlsm", "xlsb"])

if uploaded_file:
    df = load_data(uploaded_file)
    st.write("Data description:")
    st.write("Original data:", df)

    sample_data = st.checkbox("Enable data sampling")
    if sample_data:
        percentage = st.slider("Percentage", min_value=1, max_value=100, value=10)
        df = df.sample(frac=percentage * 0.01)
    st.write("Analysis results:")

    numeric_columns = [col for col in df.columns if "ID" not in col.upper() and pd.api.types.is_numeric_dtype(df[col])]
    describe_result = df[numeric_columns].describe().transpose()
    sums = df[numeric_columns].sum()
    variances = df[numeric_columns].var()

    describe_result['sum'] = sums
    describe_result['var'] = variances

    st.write(describe_result.transpose())

what_to_use = st.sidebar.radio("What to use:", options=["OpenAI", "Local model", "Hugging Face"])

if what_to_use == "OpenAI":
    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    if st.sidebar.button("Connect"):
        st.session_state.llm = ChatOpenAI(
            temperature=0, model="gpt-3.5-turbo", openai_api_key=openai_api_key
        )

if what_to_use == "Hugging Face":
    huggingface_model = st.sidebar.text_input("Model Name", placeholder="e.g. meta-llama/Meta-Llama-3-8B-Instruct")
    huggingface_api_key = st.sidebar.text_input("Hugging Face API Key", type="password")

    API_URL = f"https://api-inference.huggingface.co/models/{huggingface_model}"
    headers = {"Authorization": f"Bearer {huggingface_api_key}"}
    if st.sidebar.button("Connect Hugging Face"):
        def query(payload):
            response = requests.post(API_URL, headers=headers, json=payload)
            return response.json()


        st.session_state.hf_query = query
        st.session_state.llm = None  # No direct LLM integration for Hugging Face in LangChain yet

if what_to_use == "Local model":
    local_model_path = st.sidebar.text_input("Local model path", help="Loader used LlamaCpp.",
                                             placeholder="models/my_model.gguf")
    gpu_layers = st.sidebar.number_input("GPU layer count", min_value=-1, value=8, step=1)
    ctx_length = st.sidebar.number_input("Context length", min_value=512, value=2048, step=1024)
    max_tokens = st.sidebar.number_input("Max token count", min_value=1, value=512, step=256)

    if st.sidebar.button("Load"):
        st.session_state.llm = LlamaCpp(
            model_path=local_model_path,
            n_gpu_layers=gpu_layers,
            n_ctx=ctx_length,
            max_tokens=max_tokens,
            verbose=True,
            f16_kv=True,
            n_batch=512
        )

if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you with that data?"}]

for msg in st.session_state.messages:
    if msg["role"] == "assistant":
        st.chat_message(msg["role"]).write(msg["content"])
    elif msg["role"] == "user":
        st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("What is this data about?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if 'llm' in st.session_state and st.session_state.llm:
        llm = st.session_state.llm
        pandas_df_agent = create_pandas_dataframe_agent(
            llm,
            df,  # Passing DataFrame as input
            verbose=True,
            handle_parsing_errors=True  # Added to handle parsing errors
        )

        try:
            response = pandas_df_agent.invoke(st.session_state.messages)
            st.session_state.messages.append({"role": "assistant", "content": response['output']})
            st.chat_message("assistant").write(response["output"])

            # Extracting plot instructions from LLM output and generating plot
            if "scatter plot" in response['output'].lower():
                plt.figure(figsize=(10, 6))
                sns.scatterplot(data=df, x=numeric_columns[0], y=numeric_columns[1])
                plt.title("Scatter Plot")
                st.pyplot(plt)

            elif "line plot" in response['output'].lower():
                plt.figure(figsize=(10, 6))
                sns.lineplot(data=df, x=numeric_columns[0], y=numeric_columns[1])
                plt.title("Line Plot")
                st.pyplot(plt)

            elif "bar plot" in response['output'].lower():
                plt.figure(figsize=(10, 6))
                sns.barplot(data=df, x=numeric_columns[0], y=numeric_columns[1])
                plt.title("Bar Plot")
                st.pyplot(plt)

        except Exception as e:
            st.error(f"Could not analyze data: {e}")

    elif 'hf_query' in st.session_state:
        query = st.session_state.hf_query
        payload = {"inputs": prompt}

        try:
            response = query(payload)
            generated_text = response[0]['generated_text']
            st.session_state.messages.append({"role": "assistant", "content": generated_text})
            st.chat_message("assistant").write(generated_text)
        except Exception as e:
            st.error(f"Could not analyze data: {e}")

    else:
        st.info("Model is not loaded.")
        st.stop()
