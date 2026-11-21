import asyncio
import time
from contextlib import AsyncExitStack
import streamlit as st

from cache import buscar_cache, gravar_cache, iniciar_cache
from chat import (
    BASE_CONHECIMENTO,
    CACHE,
    MODELO,
    MODELO_CACHE,
    NOME_CACHE,
    NOME_ESPECIALISTA,
    URL_FERRAMENTAS,
    carregar_conhecimento,
    conectar_ferramentas,
    executar_ferramentas,
    iniciar_persona_assistente,
)
from llm import iniciar_llm, obter_resposta

st.set_page_config(
    page_title=f"{NOME_ESPECIALISTA} — Especialista em Suplementação",
    page_icon="💪",
    layout="wide",
)

# carregar_infraestrutura() carrega o modelo de embeddings, o ChromaDB e o JSON uma única vez em memória para evitar lentidão a cada clique.
@st.cache_resource(show_spinner="Carregando modelo de embeddings e cache semântico...")
def carregar_infraestrutura():
    iniciado, modelo_cache, colecao_cache = iniciar_cache(MODELO_CACHE, CACHE, NOME_CACHE)
    if not iniciado:
        raise RuntimeError("Não foi possível iniciar o cache semântico (ChromaDB).")

    carregado, conhecimento = asyncio.run(carregar_conhecimento(BASE_CONHECIMENTO))
    if not carregado:
        raise RuntimeError("Não foi possível carregar a base de conhecimento (suplementos.json).")

    return modelo_cache, colecao_cache, conhecimento

# _verificar_mcp_async() e verificar_status_mcp() (os dois abaixo) testam a conexão com o servidor MCP na porta 8000 para listar as ferramentas ativas na barra lateral
async def _verificar_mcp_async():
    async with AsyncExitStack() as contexto:
        _, ferramentas = await conectar_ferramentas(contexto, URL_FERRAMENTAS)
        return [f["name"] for f in ferramentas]

def verificar_status_mcp():
    try:
        return asyncio.run(_verificar_mcp_async())
    except Exception:
        return []

# _processar_com_llm_async(persona) envia o histórico ao Gemini e resolve o ciclo de chamadas às ferramentas MCP até obter a resposta final
async def _processar_com_llm_async(persona):
    iniciado, llm = await iniciar_llm()
    if not iniciado:
        return None, [], "Erro ao iniciar conexão com o Gemini. Verifique o arquivo .env."

    async with AsyncExitStack() as contexto:
        conexao_mcp, ferramentas = await conectar_ferramentas(contexto, URL_FERRAMENTAS)
        ferramentas_usadas = []
        resposta_final = None

        while True:
            sucesso, resposta = await obter_resposta(llm, persona, ferramentas, modelo=MODELO)

            if not sucesso or resposta is None:
                return None, ferramentas_usadas, "Não foi possível obter resposta do Gemini. Tente novamente."

            persona.extend(resposta.steps)

            chamadas = [
                passo.name
                for passo in resposta.steps
                if getattr(passo, "type", None) == "function_call"
            ]

            if chamadas and conexao_mcp is not None:
                ferramentas_usadas.extend(chamadas)
                await executar_ferramentas(conexao_mcp, resposta, persona)
            else:
                resposta_final = resposta.output_text
                break

        return resposta_final, ferramentas_usadas, None

# inicializar_sessao(conhecimento) cria as variáveis de estado da sessão do usuário no Streamlit (histórico, persona e contadores de telemetria)
def inicializar_sessao(conhecimento):
    if "persona" not in st.session_state:
        st.session_state.persona = iniciar_persona_assistente(conhecimento)
    if "mensagens" not in st.session_state:
        st.session_state.mensagens = [
            {
                "role": "assistant",
                "content": (
                    f"Olá! Eu sou o **{NOME_ESPECIALISTA}**, um sistema especialista em "
                    "suplementação esportiva e nutrição. Como posso te ajudar hoje?"
                ),
                "origem": None,
                "tempo": None,
            }
        ]
    if "metricas" not in st.session_state:
        st.session_state.metricas = {"total": 0, "cache": 0, "mcp": 0, "llm": 0}
    if "ferramentas_mcp" not in st.session_state:
        st.session_state.ferramentas_mcp = verificar_status_mcp()
    if "prompt_sugerido" not in st.session_state:
        st.session_state.prompt_sugerido = None

# responder_usuario() processa uma mensagem medindo o tempo de resposta, priorizando o cache semântico antes de consultar o LLM/MCP
def responder_usuario(mensagem, modelo_cache, colecao_cache):
    st.session_state.mensagens.append(
        {"role": "user", "content": mensagem, "origem": None, "tempo": None}
    )
    st.session_state.persona.append(
        {"type": "user_input", "content": [{"type": "text", "text": mensagem}]}
    )
    st.session_state.metricas["total"] += 1

    inicio = time.perf_counter()

    # 1. Tenta recuperar do Cache Semântico
    encontrada, resposta_cache = buscar_cache(mensagem, modelo_cache, colecao_cache)
    if encontrada:
        tempo = time.perf_counter() - inicio
        st.session_state.persona.append({"type": "model_output", "text": resposta_cache})
        st.session_state.metricas["cache"] += 1
        st.session_state.mensagens.append(
            {
                "role": "assistant",
                "content": resposta_cache,
                "origem": "⚡ Cache Semântico",
                "tempo": tempo,
            }
        )
        return

    # 2. Caso não esteja no cache, consulta LLM + MCP
    resposta_final, ferramentas_usadas, erro = asyncio.run(
        _processar_com_llm_async(st.session_state.persona)
    )
    tempo = time.perf_counter() - inicio

    if erro:
        st.session_state.mensagens.append(
            {
                "role": "assistant",
                "content": f"❌ {erro}",
                "origem": "Erro",
                "tempo": tempo,
            }
        )
        return

    if ferramentas_usadas:
        nomes_ferramentas = ", ".join(dict.fromkeys(ferramentas_usadas))
        origem = f"🛠️ MCP ({nomes_ferramentas})"
        st.session_state.metricas["mcp"] += 1
    else:
        origem = "🧠 LLM (Gemini)"
        st.session_state.metricas["llm"] += 1
        gravar_cache(mensagem, resposta_final, modelo_cache, colecao_cache)

    st.session_state.mensagens.append(
        {
            "role": "assistant",
            "content": resposta_final,
            "origem": origem,
            "tempo": tempo,
        }
    )

# main() monta a interface web com barra lateral (status MCP, métricas e atalhos de teste) e área principal de chat
def main():
    try:
        modelo_cache, colecao_cache, conhecimento = carregar_infraestrutura()
    except Exception as e:
        st.error(f"Erro ao iniciar o {NOME_ESPECIALISTA}: {e}")
        st.stop()

    inicializar_sessao(conhecimento)

    # --- Barra Lateral (Dashboard e Controles) ---
    with st.sidebar:
        st.title(f"💪 {NOME_ESPECIALISTA}")
        st.caption("Sistema Especialista em Suplementação Esportiva")

        st.divider()
        st.subheader("🔌 Servidor MCP")
        ferramentas_ativas = st.session_state.ferramentas_mcp
        if ferramentas_ativas:
            st.success(f"Online ({len(ferramentas_ativas)} ferramentas)")
            for nome in ferramentas_ativas:
                st.markdown(f"- `{nome}`")
        else:
            st.warning("Offline — inicie `python ferramentas.py`")

        if st.button("🔄 Atualizar status MCP", use_container_width=True):
            st.session_state.ferramentas_mcp = verificar_status_mcp()
            st.rerun()

        st.divider()
        st.subheader("📊 Telemetria da Sessão")
        m = st.session_state.metricas
        col1, col2 = st.columns(2)
        col1.metric("Perguntas", m["total"])
        col2.metric("⚡ Cache", m["cache"])
        col3, col4 = st.columns(2)
        col3.metric("🛠️ MCP", m["mcp"])
        col4.metric("🧠 LLM", m["llm"])

        if m["total"] > 0:
            taxa_cache = (m["cache"] / m["total"]) * 100
            st.caption(f"Taxa de economia por cache: **{taxa_cache:.0f}%**")

        st.divider()
        st.subheader("🧪 Testes Rápidos")
        sugestoes = [
            "O que é hipertrofia muscular?",
            "Quais os benefícios da beta-alanina?",
            "Peso 75 kg, qual minha dose de creatina?",
            "Quem descobriu o Brasil?",
        ]
        for texto in sugestoes:
            if st.button(texto, use_container_width=True):
                st.session_state.prompt_sugerido = texto

        st.divider()
        if st.button("🗑️ Limpar conversa", use_container_width=True):
            st.session_state.persona = iniciar_persona_assistente(conhecimento)
            st.session_state.mensagens = st.session_state.mensagens[:1]
            st.session_state.metricas = {"total": 0, "cache": 0, "mcp": 0, "llm": 0}
            st.rerun()

        st.caption(
            "⚠️ Aviso: orientações informativas. Não substitui acompanhamento "
            "médico ou nutricional."
        )

    # --- Área Principal do Chat ---
    st.header(f"🏋️ {NOME_ESPECIALISTA} — Assistente de Suplementação")
    st.caption(
        "Motor LLM (Gemini) + Ferramentas MCP + Cache Semântico (ChromaDB)"
    )

    for msg in st.session_state.mensagens:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["origem"] and msg["tempo"] is not None:
                st.caption(f"{msg['origem']}  •  ⏱️ {msg['tempo']:.2f}s")

    entrada_usuario = st.chat_input("Digite sua dúvida sobre suplementação esportiva...")
    if st.session_state.prompt_sugerido:
        entrada_usuario = st.session_state.prompt_sugerido
        st.session_state.prompt_sugerido = None

    if entrada_usuario:
        with st.chat_message("user"):
            st.markdown(entrada_usuario)

        with st.spinner("Consultando o SupleMentor..."):
            responder_usuario(entrada_usuario, modelo_cache, colecao_cache)

        st.rerun()

if __name__ == "__main__":
    main()