import langchain_core
import requests
from langchain.agents import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import LlamaCpp
import streamlit as st
import pandas as pd
import os

file_formats = {
    "csv": pd.read_csv,
    "xls": pd.read_excel,
    "xlsx": pd.read_excel,
    "xlsm": pd.read_excel,
    "xlsb": pd.read_excel,
}

@st.cache(ttl=3600)
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

st.set_page_config(page_title="DataFrame Analysis with LLM", page_icon="🤖")

st.title("DataFrame Analysis using Large Language Models.")

st.sidebar.title("Settings")

enable_summarization = False
if st.sidebar.checkbox("Enable summarization before sending data to LLM", help="Automatically creates another table with summary of the data and sends it to LLM on each request as an additional input."):
    enable_summarization = True

sampling_percentage = 100
sample_data = st.sidebar.checkbox("Enable data sampling")
if sample_data:
    sampling_percentage = st.sidebar.slider("Sampling percentage", min_value=1, max_value=100, value=10, help="How much of the original data is sent to LLM.")


uploaded_file = st.file_uploader("Load CSV or Excel file", type=["csv", "xls", "xlsx", "xlsm", "xlsb"])

if uploaded_file:
    final_dataframe = pd.read_csv(uploaded_file)
    st.write("Data description:")
    st.write("Original data:", final_dataframe)

    final_dataframe = final_dataframe.sample(frac=sampling_percentage * 0.01)

    dataframes_to_send = [final_dataframe]

    if enable_summarization:
        st.write("Analysis results:")

        numeric_columns = [col for col in final_dataframe.columns if "ID" not in col.upper() and pd.api.types.is_numeric_dtype(final_dataframe[col])]
        general_analysis_report_of_dataframe = final_dataframe[numeric_columns].describe().transpose()
        sums = final_dataframe[numeric_columns].sum()
        variances = final_dataframe[numeric_columns].var()

        general_analysis_report_of_dataframe['sum'] = sums
        general_analysis_report_of_dataframe['var'] = variances

        st.write(general_analysis_report_of_dataframe.transpose())
        dataframes_to_send.append(general_analysis_report_of_dataframe)

what_to_use = st.sidebar.radio("What to use:", options=["OpenAI", "Local model","Hugging Face"])

if what_to_use == "OpenAI":
    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

    model_name = st.sidebar.radio("Model:", options=["gpt-3.5-turbo", "gpt-4-turbo", "gpt-4o"])

    if st.sidebar.button("Connect"):
        st.session_state.llm = ChatOpenAI(
            temperature=0, model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, streaming=True
        )
if what_to_use == "Hugging Face":
    huggingface_model = st.sidebar.text_input("Model Name",placeholder = "e.g. meta-llama/Meta-Llama-3-8B-Instruct")
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
    st.sidebar.info("Tested local model: CommandR+")

    local_model_path = st.sidebar.text_input("Local model path",
                                             help="Loader used LlamaCpp.",
                                             placeholder="models/my_model.gguf")

    st.sidebar.warning("Keep in mind that GPU settings will only work if llama-cpp-python is installed with GPU acceleration enabled (BLAS = 1).")

    gpu_layers = st.sidebar.number_input("GPU layer count",
                                         min_value=-1,
                                         value=8,
                                         step=1,
                                         help="The more layers are offset to GPU, the faster the model should generate."
                                              "Set to 0 if you do not have dedicated GPU.")

    ctx_length = st.sidebar.number_input("Context length",
                                         min_value=512,
                                         value=2048,
                                         step=1024,
                                         help="The higher the context length - the more the model will remember, "
                                              "however at a cost of higher memory usage.")

    max_tokens = st.sidebar.number_input("Max token count",
                                         min_value=1,
                                         value=512,
                                         step=256,
                                         help="How many tokens AI should generate, keep in mind that in data analysis,"
                                         "majority of the tokens are used for thought generation, that is not actually"
                                         "given to the user as an output, so this value should be A LOT higher than in"
                                         "most use cases.")

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
            dataframes_to_send,  # Passing DataFrame(s) as input
            verbose=True,
            handle_parsing_errors=True
        )

        try:
            response = pandas_df_agent.invoke(st.session_state.messages)
            st.session_state.messages.append({"role": "assistant", "content": response['output']})
            st.chat_message("assistant").write(response["output"])
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