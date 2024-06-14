import langchain_core
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

st.set_page_config(page_title="DataFrame analysis")

#st.markdown(
#    """
#    <style>
#    .stApp {
#        background: url("https://media.licdn.com/dms/image/C5112AQGWCENbwjTCLw/article-cover_image-shrink_600_2000/0/1520095639394?e=2147483647&v=beta&t=l046v8DL2uB4B-mHN-731BhHK0OcNkh47NztypL1KHI") no-repeat center center fixed !important;
#        background-size: cover !important;
#    }
#    .main {
#        background: none !important;
#        padding-left: 10% !important;
#        padding-right: 40% !important;
#    }
#    </style>
#    """,
#    unsafe_allow_html=True
#)

# def query(payload):
#     headers = {"Authorization": f"Bearer {API_KEY}"}
#     response = requests.post(API_URL, headers=headers, json=payload)
#     try:
#         response_data = response.json()
#         generated_text = response_data[0]['generated_text']
#         return generated_text
#     except Exception as e:
#         st.error(f"Nie udało się pobrać odpowiedzi: {e}")
#         return None

st.title("DataFrame Analysis")

st.sidebar.write("""
How to use:
1. Load CSV or Excel file.
3. Pick your sampling method.
4. Ask LLM about the data.
""")

uploaded_file = st.file_uploader("Load CSV or Excel file", type=["csv", "xls", "xlsx", "xlsm", "xlsb"])

# if uploaded_file:
#     data_frame = load_data(uploaded_file)
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Data description:")
    st.write("Original data:", df)

    sample_data = st.checkbox("Enable data sampling")
    if sample_data:
        percentage = st.slider("Percentage", min_value=1, max_value=100, value=10)
        df = df.sample(frac=percentage * 0.01)
        # result_df = analyze_data(df, sample=sample_data)
    st.write("Analysis results:")

    numeric_columns = [col for col in df.columns if "ID" not in col.upper() and pd.api.types.is_numeric_dtype(df[col])]
    describe_result = df[numeric_columns].describe().transpose()
    sums = df[numeric_columns].sum()
    variances = df[numeric_columns].var()

    describe_result['sum'] = sums
    describe_result['var'] = variances

    st.write(describe_result.transpose())


    # for msg in st.session_state.messages:
    #     if msg["role"] == "assistant":
    #         st.text(f"Assistant: {msg['content']}")
    #     elif msg["role"] == "user":
    #         st.text(f"User: {msg['content']}")

    # if prompt := st.text_input("O czym są te dane?"):
    #     st.session_state.messages.append({"role": "user", "content": prompt})
what_to_use = st.sidebar.radio("What to use:", options=["OpenAI", "Local model"])

if what_to_use == "OpenAI":
    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

    if st.sidebar.button("Connect"):
        st.session_state.llm = ChatOpenAI(
            temperature=0, model="gpt-3.5-turbo", openai_api_key=openai_api_key, streaming=True
        )

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
    if 'llm' in st.session_state:
        llm = st.session_state.llm
    else:
        st.info("Model is not loaded.")
        st.stop()
    if 'df' not in locals():
        st.info("Data file is not loaded.")
        st.stop()

    pandas_df_agent = create_pandas_dataframe_agent(
        llm,
        df,  # Teraz przekazujemy DataFrame jako dane wejściowe
        verbose=True,
        handle_parsing_errors=True
    )

    try:
        response = pandas_df_agent.invoke(st.session_state.messages)
    except langchain_core.exceptions.OutputParserException as e:
        st.error(f"Could not analyze data: {e}")
        st.stop()

    st.session_state.messages.append({"role": "assistant", "content": response['output']})
    st.chat_message("assistant").write(response["output"])

