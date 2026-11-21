import json
import unicodedata
from pathlib import Path
from mcp.server.mcpserver import MCPServer

NOME = "analisador_suplementacao"
mcp = MCPServer(NOME, "Motor de regras especialista para consulta, segurança e cálculo de suplementação esportiva")

# CAMINHO_TABELA, MAPA_CONDICOES sao os caminhos da base estruturada e termos usados para identificar condições de saúde na pergunta
CAMINHO_TABELA = Path(__file__).resolve().parent / "conhecimento" / "tabela_suplementos.json"

MAPA_CONDICOES = {
    "hipertensao": ["hipertens", "pressao alta", "cardiac", "coracao", "arritmia", "taquicardia"],
    "insonia": ["insonia", "sono", "dormir", "ansiedade", "ansioso", "ansiosa", "panico"],
    "lactose": ["lactose", "leite", "intoleran", "aplv"],
    "renal": ["renal", "rim", "rins", "creatinina", "nefro"],
    "gestacao": ["gravida", "gestante", "gravidez", "lactante", "amament"]
}

# _normalizar(texto) remove acentos e padroniza em minúsculas para facilitar a comparação de nomes
def _normalizar(texto: str) -> str:
    texto_sem_acento = unicodedata.normalize("NFKD", str(texto)).encode("ASCII", "ignore").decode("ASCII")
    return texto_sem_acento.lower().strip()


# _carregar_tabela() lê a ficha técnica dos suplementos direto do JSON para manter os dados separados do código
def _carregar_tabela():
    with open(CAMINHO_TABELA, "r", encoding="utf-8") as arquivo:
        return json.load(arquivo)

# _buscar_item(nome_suplemento)procura primeiro por correspondência exata (nome ou apelido) e depois por busca parcial
def _buscar_item(nome_suplemento: str):
    termo = _normalizar(nome_suplemento)
    tabela = _carregar_tabela()

    # 1. Busca exata por alias ou nome
    for item in tabela:
        aliases = [_normalizar(a) for a in item.get("aliases", [])]
        if termo == _normalizar(item["nome"]) or termo in aliases:
            return item

    # 2. Busca parcial (substring)
    for item in tabela:
        aliases = [_normalizar(a) for a in item.get("aliases", [])]
        if any(a in termo or termo in a for a in aliases):
            return item

    return None

# consultar_suplemento(nome_suplemento) é uma ferramenta MCP, retorna benefícios, dose padrão, horário e restrições gerais do suplemento
@mcp.tool(
    name="consultar_suplemento",
    title="Consulta Técnica de Suplemento",
    description="Consulta ficha técnica estruturada (benefícios, dosagem padrão, melhor horário e restrições) de um suplemento esportivo."
)
def consultar_suplemento(nome_suplemento: str) -> str:
    item = _buscar_item(nome_suplemento)
    if not item:
        disponiveis = ", ".join(s["nome"] for s in _carregar_tabela())
        return (
            f"Suplemento '{nome_suplemento}' não encontrado na base estruturada. "
            f"Suplementos cadastrados: {disponiveis}."
        )

    return (
        f"Ficha Técnica — {item['nome']}:\n"
        f"- Benefícios: {item['beneficios']}\n"
        f"- Dosagem padrão: {item['dosagem_padrao']}\n"
        f"- Melhor horário: {item['melhor_horario']}\n"
        f"- Restrições gerais: {item['restricoes']}"
    )

# verificar_seguranca(nome_suplemento, condicoes_usuario), outra ferramenta MCP, cruza o suplemento com condições de saúde informadas e aplica regras de contraindicação
@mcp.tool(
    name="verificar_seguranca",
    title="Verificador de Contraindicações e Segurança",
    description=(
        "Avalia regras de segurança e contraindicações de um suplemento para usuários com condições específicas "
        "(ex: hipertensão, intolerância à lactose, problema renal, insônia, ansiedade, gestação)."
    )
)
def verificar_seguranca(nome_suplemento: str, condicoes_usuario: str) -> str:
    item = _buscar_item(nome_suplemento)
    if not item:
        return (
            f"Não há ficha de segurança cadastrada para '{nome_suplemento}'. "
            "Recomende sempre avaliação com médico ou nutricionista antes do uso."
        )

    texto_condicoes = _normalizar(condicoes_usuario)
    condicoes_identificadas = []

    for chave_condicao, palavras_chave in MAPA_CONDICOES.items():
        if any(p in texto_condicoes for p in palavras_chave):
            condicoes_identificadas.append(chave_condicao)

    regras = item.get("regras_seguranca", {})
    alertas = []
    for cond in condicoes_identificadas:
        if cond in regras:
            alertas.append(f"[{cond.upper()}] {regras[cond]}")

    aviso_padrao = "Importante: esta análise é informativa e não substitui avaliação médica ou nutricional."

    if alertas:
        return (
            f"Análise de Segurança para {item['nome']} (condição informada: {condicoes_usuario}):\n"
            + "\n".join(alertas)
            + f"\n{aviso_padrao}"
        )

    return (
        f"Análise de Segurança para {item['nome']} (condição informada: {condicoes_usuario}):\n"
        f"Nenhuma contraindicação crítica mapeada na base para a condição descrita. "
        f"Restrição geral do suplemento: {item['restricoes']} {aviso_padrao}"
    )

# calcular_dose_creatina(peso_kg), mais uma ferramenta MCP, calcula a faixa diária de creatina (0,05g a 0,07g por kg) validando limites de peso
@mcp.tool(
    name="calcular_dose_creatina",
    title="Calculadora de Creatina",
    description="Calcula a dose diária recomendada de creatina com base no peso corporal do usuário em kg."
)
def calcular_dose_creatina(peso_kg: float) -> str:
    try:
        peso = float(str(peso_kg).replace(",", "."))
        if peso <= 20 or peso > 250:
            return "Peso fora da faixa válida (informe um peso em kg entre 20 e 250)."

        dose_minima = peso * 0.05
        dose_maxima = peso * 0.07
        return (
            f"Para {peso:.1f} kg, a faixa personalizada de creatina é de "
            f"{dose_minima:.2f}g a {dose_maxima:.2f}g por dia (referência geral: 3g a 5g/dia, uso contínuo)."
        )
    except ValueError:
        return "Erro: informe um peso numérico válido em quilogramas."

# calcular_meta_proteina(peso_kg, objetivo),ferramenta MCP, estima a meta diária de proteína em gramas conforme o peso e o objetivo de treino
@mcp.tool(
    name="calcular_meta_proteina",
    title="Calculadora de Meta Proteica Diária",
    description=(
        "Calcula a meta diária de proteínas em gramas com base no peso corporal (kg) e objetivo de treino "
        "(hipertrofia, emagrecimento, resistencia/endurance ou manutencao)."
    )
)
def calcular_meta_proteina(peso_kg: float, objetivo: str) -> str:
    try:
        peso = float(str(peso_kg).replace(",", "."))
        if peso <= 20 or peso > 250:
            return "Peso fora da faixa válida (informe um peso em kg entre 20 e 250)."

        obj = _normalizar(objetivo)
        if "hipertrof" in obj or "massa" in obj or "ganho" in obj:
            fator_min, fator_max, rotulo = 1.6, 2.2, "Hipertrofia"
        elif "emagrec" in obj or "defin" in obj or "cutting" in obj or "gordura" in obj:
            fator_min, fator_max, rotulo = 1.8, 2.2, "Emagrecimento / Preservação de massa magra"
        elif "resist" in obj or "endurance" in obj or "corrida" in obj or "ciclismo" in obj:
            fator_min, fator_max, rotulo = 1.2, 1.6, "Resistência / Endurance"
        else:
            fator_min, fator_max, rotulo = 1.2, 1.6, "Manutenção / Saúde geral"

        g_min = peso * fator_min
        g_max = peso * fator_max
        return (
            f"Meta proteica diária para {peso:.1f} kg (objetivo: {rotulo}): "
            f"entre {g_min:.0f}g e {g_max:.0f}g de proteína por dia "
            f"({fator_min:.1f}g a {fator_max:.1f}g/kg), somando alimentação sólida e suplementos como Whey Protein."
        )
    except ValueError:
        return "Erro: informe um peso numérico válido em quilogramas."
    

# if __name__ == "__main__", inicia o servidor MCP via HTTP na porta 8000 na rota /mcp
if __name__ == "__main__":
    # --- TESTE MANUAL ---
    # print(consultar_suplemento("whey concentrado"))
    # print(verificar_seguranca("termogênico", "tenho hipertensão"))
    # print(verificar_seguranca("whey concentrado", "sou intolerante à lactose"))
    # print(calcular_dose_creatina(75))
    # print(calcular_meta_proteina(75, "hipertrofia"))
    # ------------------------------

    mcp.run(transport="streamable-http", host="0.0.0.0", streamable_http_path="/mcp")
