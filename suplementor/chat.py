from mcp.client.streamable_http import streamable_http_client
from contextlib import AsyncExitStack
from mcp import ClientSession
from cache import *
from llm import *
import asyncio
import json

# Constantes de configuração (MODELO, URL_FERRAMENTAS, etc.) sao parâmetros globais do modelo Gemini, endereço do servidor MCP e arquivos de base/cache
NOME_ESPECIALISTA = "SupleMentor"
BASE_CONHECIMENTO = "conhecimento/suplementos.json"
MODELO = "gemini-3.5-flash-lite"

CACHE = "cache"
MODELO_CACHE = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
NOME_CACHE = "suplementor_cache"

URL_FERRAMENTAS = "http://localhost:8000/mcp"

def iniciar_persona_assistente(conhecimento):
    persona = [
        {
            "type": "user_input",
            "content": [{ "type": "text", "text": f"Você é um especialista em suplementação esportiva e nutrição, chamado {NOME_ESPECIALISTA}."}]
        },
        {
            "type": "user_input",
            "content": [{ "type": "text", "text": "Se o usuário fizer uma saudação, você deve falar na sua resposta que você é um software de IA, mas que está pronto para tirar dúvidas sobre suplementos e treinos."}]
        },
        {
            "type": "user_input",
            "content": [{ "type": "text", "text": "GUARDA-RAIL: Se o usuário te perguntar algo que não esteja relacionado a suplementação, esportes, nutrição ou bem-estar, você DEVE recusar educadamente dizendo 'não posso responder esta pergunta, meu foco é apenas suplementação e esportes'."}]
        },
        {
            "type": "user_input",
            "content": [{ "type": "text", "text": "Sempre que o usuário perguntar sobre um suplemento específico (ex: whey, creatina, bcaa), você deve usar a ferramenta 'consultar_suplemento' para buscar informações na base de dados."}]
        },
        {
            "type": "user_input",
            "content": [{ "type": "text", "text": "Se o usuário informar o peso dele e perguntar sobre creatina, você deve obrigatoriamente executar a ferramenta 'calcular_dose_creatina'."}]
        },
        {
            "type": "user_input",
            "content": [{ "type": "text", "text": "Sempre que o usuário mencionar qualquer condição de saúde ou restrição (como hipertensão, pressão alta, intolerância à lactose, problema renal, insônia, ansiedade ou gravidez) associada a um suplemento, você DEVE obrigatoriamente chamar a ferramenta 'verificar_seguranca' antes de responder."}]
        },
        {
            "type": "user_input",
            "content": [{ "type": "text", "text": "Se o usuário informar o peso dele e perguntar quanta proteína deve consumir por dia (para hipertrofia, emagrecimento, resistência ou manutenção), você deve obrigatoriamente usar a ferramenta 'calcular_meta_proteina'."}]
        },
    ]

    for mensagem, resposta in conhecimento:
        persona.append({
            "type": "user_input",
            "content": [{ "type": "text", "text": f"Caso o usuário fale ou pergunte '{mensagem}' ou algo parecido, sua resposta deve ser baseada nesta diretriz: '{resposta}'"}]
        })

    return persona

# carregar_conhecimento() lê o JSON de perguntas e respostas usado como referência de domínio no prompt
async def carregar_conhecimento(base_conhecimento = BASE_CONHECIMENTO):
    carregado, conhecimento = False, []

    try:
        with open(base_conhecimento, "r", encoding = "utf-8") as arquivo:
            conteudo = json.load(arquivo)

        for pergunta_resposta in conteudo:
            pergunta = pergunta_resposta["pergunta"]
            resposta = pergunta_resposta["resposta"]

            conhecimento.append((pergunta, resposta))

        carregado = True
    except Exception as e:
        print(f"erro carregando conhecimento: {e}")

    return carregado, conhecimento

# conectar_ferramentas() conecta ao servidor MCP via Streamable HTTP e converte as ferramentas para o formato esperado pelo Gemini
async def conectar_ferramentas(contexto, url_ferramentas = URL_FERRAMENTAS):
    conexao_mcp, ferramentas = None, [] 

    try:
        stream_leitura, stream_escrita = await contexto.enter_async_context(streamable_http_client(url_ferramentas))

        conexao_mcp = await contexto.enter_async_context(ClientSession(stream_leitura, stream_escrita))

        await conexao_mcp.initialize()

        listagem_ferramentas = await conexao_mcp.list_tools()
        for f in listagem_ferramentas.tools:
            ferramentas.append({
                "type": "function",
                "name": f.name,
                "description": f.description,
                "parameters": f.input_schema
            })
    except Exception as e:
        print(f"erro conectando as ferramentas: {e}")

    return conexao_mcp, ferramentas

def extrair_texto_execucao(resultado):
    if resultado.structured_content:
        return json.dumps(resultado.structured_content, ensure_ascii = False)

    conteudo = []
    for c in resultado.content:
        if hasattr(c, "text"):
            conteudo.append(c.text)
        else:
            conteudo.append(str(c))

    return "\n".join(conteudo)

# executar_ferramentas() intercepta chamadas de função pedidas pelo modelo, executa no servidor MCP e devolve o resultado ao contexto
async def executar_ferramentas(conexao_mcp, resposta, mensagens):
    ferramentas_para_executar = [passo for passo in resposta.steps if passo.type == "function_call"]
    if ferramentas_para_executar:
        for ferramenta in ferramentas_para_executar:
            execucao = await conexao_mcp.call_tool(ferramenta.name, arguments = ferramenta.arguments)

            mensagens.append({
                "type": "function_result",
                "name": ferramenta.name,
                "call_id": ferramenta.id,
                "result": [
                    {
                        "type": "text",
                        "text": extrair_texto_execucao(execucao)
                    }
                ]
            })

    return ferramentas_para_executar != None and len(ferramentas_para_executar) > 0

# executar() é o loop principal do chat no terminal, checa o cache semântico antes de chamar o LLM e só grava no cache se não usar MCP
async def executar():
    contexto = AsyncExitStack()
    conexao_mcp, ferramentas = await conectar_ferramentas(contexto)
    if not ferramentas:
        print("não foram encontradas ferramentas MCP. Inicie ferramentas.py primeiro.")
    else:
        for ferramenta in ferramentas:
            print(f"ferramenta MCP conectada: {ferramenta['name']}")

    iniciado, llm = await iniciar_llm()
    if not iniciado:
        await contexto.aclose()
        print("erro iniciando conexão com o Gemini")
        return    

    carregado, conhecimento = await carregar_conhecimento()
    if not carregado:
        await contexto.aclose()
        print("erro carregando conhecimento (suplementos.json)")
        return    

    iniciado, modelo_cache, colecao_cache = iniciar_cache(MODELO_CACHE, CACHE, NOME_CACHE)
    if not iniciado:
        await contexto.aclose()
        print("erro iniciando cache vetorial")
        return    

    persona = iniciar_persona_assistente(conhecimento)
    print("\n✅ SupleMentor iniciado com sucesso! Digite 'sair' para encerrar.\n")

    while True:
        mensagem = input("👤 Você: ")
        if mensagem.lower() in ["sair", "/s"]:
            await contexto.aclose()
            break

        persona.append({
            "type": "user_input",
            "content": [{ "type": "text", "text": mensagem }]
        })

        #encontrada, resposta = buscar_cache(mensagem, modelo_cache, colecao_cache)
        #if encontrada:
        #    persona.append({ "type": "model_output", "text": resposta})
        #    print(f"🤖 SupleMentor: {resposta}\n[⚡ Recuperada do Cache Semântico]\n")
        #else:
        #    ferramentas_executadas = False

        #    while True:
        #        sucesso, resposta = await obter_resposta(llm, persona, ferramentas, modelo = MODELO)
        #        persona.extend(resposta.steps)                

        #        if sucesso:
        #            # Ocultando os logs de tokens para um chat mais limpo
        #            # print(f"🤖 tokens consumidos: {get_consumo_tokens(resposta)}")
                    
        #            if not await executar_ferramentas(conexao_mcp, resposta, persona):
        #                resposta = resposta.output_text
        #                break
        #            else:
        #                ferramentas_executadas = True

        #    print(f"🤖 SupleMentor: {resposta}")
        #    if ferramentas_executadas:
        #         print(f"[🛠️ Resposta baseada em consulta MCP]\n")
        #    else:
        #         print(f"[🧠 Resposta processada pelo LLM]\n")
                 
        #   if not ferramentas_executadas:
        #       gravar_cache(mensagem, resposta, modelo_cache, colecao_cache)
        encontrada, resposta = buscar_cache(mensagem, modelo_cache, colecao_cache)
        if encontrada:
            persona.append({ "type": "model_output", "text": resposta})
            print(f"🤖 SupleMentor: {resposta}\n[⚡ Recuperada do Cache Semântico]\n")
        else:
            print("⏳ Consultando o SupleMentor...")
            ferramentas_executadas = False
            resposta_final = None

            while True:
                sucesso, resposta = await obter_resposta(llm, persona, ferramentas, modelo = MODELO)

                if not sucesso or resposta is None:
                    print("❌ Não foi possível obter resposta do Gemini. Verifique sua chave no .env ou sua conexão.\n")
                    break

                persona.extend(resposta.steps)

                if not await executar_ferramentas(conexao_mcp, resposta, persona):
                    resposta_final = resposta.output_text
                    break
                else:
                    ferramentas_executadas = True

            if resposta_final:
                print(f"🤖 SupleMentor: {resposta_final}")
                if ferramentas_executadas:
                    print("[🛠️ Resposta baseada em consulta MCP]\n")
                else:
                    print("[🧠 Resposta processada pelo LLM]\n")
                    gravar_cache(mensagem, resposta_final, modelo_cache, colecao_cache)

    await contexto.aclose()

if __name__ == "__main__":
    asyncio.run(executar())