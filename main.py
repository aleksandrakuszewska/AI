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
        st.error(f"Nieobsługiwany format pliku: {ext}")
        return None

st.set_page_config(page_title="Analiza danych tabelarycznych")

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

st.title("Analiza danych tabelarycznych")

st.sidebar.write("""
### Jak używać:
1. Załaduj plik CSV lub Excel.
2. Zobacz opis danych.
3. Wybierz opcję próbkowania danych.
4. Zobacz wyniki analizy.
5. Zadawaj pytania dotyczące danych.
""")

uploaded_file = st.file_uploader("Załaduj plik CSV lub Excel", type=["csv", "xls", "xlsx", "xlsm", "xlsb"])

# if uploaded_file:
#     data_frame = load_data(uploaded_file)
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Opis danych:")
    st.write("Oryginalne dane:", df)

    sample_data = st.checkbox("Próbkować dane (20%)")
    if sample_data:
        df = df.sample(frac=0.2)
        # result_df = analyze_data(df, sample=sample_data)
    st.write("Wyniki analizy:")

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
what_to_use = st.sidebar.radio("Co wybrać:", options=["OpenAI", "Lokalny model"])

if what_to_use == "OpenAI":
    openai_api_key = st.sidebar.text_input("Klucz API OpenAI", type="password")

    if st.sidebar.button("Połącz"):
        st.session_state.llm = ChatOpenAI(
            temperature=0, model="gpt-3.5-turbo", openai_api_key=openai_api_key, streaming=True
        )

if what_to_use == "Lokalny model":
    st.sidebar.info("Testowany model lokalny: CommandR+")

    local_model_path = st.sidebar.text_input("Ścieżka do lokalnego modelu",
                                             help="Używany loader to LlamaCpp.",
                                             placeholder="models/my_model.gguf")

    st.sidebar.warning("Pamiętaj, że ustawienia GPU będą działać tylko wtedy, gdy llama-cpp-python zostanie zainstalowany z włączonym przyspieszeniem GPU.")

    gpu_layers = st.sidebar.number_input("Warstwy GPU",
                                         min_value=-1,
                                         value=8,
                                         step=1,
                                         help="Ustaw tak wysoko, jak to możliwe, aby uzyskać najszybszą odpowiedź. "
                                              "Ustaw na 0, jeśli nie masz dedykowanej karty GPU.")

    ctx_length = st.sidebar.number_input("Długość kontekstu",
                                         min_value=512,
                                         value=2048,
                                         step=1024,
                                         help="Długość kontekstu dla modelu. Im dłuższy, tym lepsza pamięć AI, "
                                              "jednak kosztem prędkości i użycia pamięci fizycznej.")

    max_tokens = st.sidebar.number_input("Maksymalna ilość tokenów",
                                         min_value=1,
                                         value=64,
                                         step=64,
                                         help="Ile tokenów AI powinno wygenerować w odpowiedzi.")

    if st.sidebar.button("Załaduj"):
        st.session_state.llm = LlamaCpp(
            model_path=local_model_path,
            n_gpu_layers=gpu_layers,
            n_ctx=ctx_length,
            max_tokens=max_tokens,
            verbose=True,
            f16_kv=True,
            n_batch=512
        )

if "messages" not in st.session_state or st.sidebar.button("Wyczyść historię rozmowy"):
    st.session_state["messages"] = [{"role": "assistant", "content": "Co chcesz wiedzieć na temat danych?"}]

for msg in st.session_state.messages:
    if msg["role"] == "assistant":
        st.chat_message(msg["role"]).write(msg["content"])
    elif msg["role"] == "user":
        st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("O czym są te dane?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    if 'llm' in st.session_state:
        llm = st.session_state.llm
    else:
        st.info("Model nie jest załadowany.")
        st.stop()
    if 'df' not in locals():
        st.info("Plik z danymi nie został wybrany.")
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

