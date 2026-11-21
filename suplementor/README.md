# SupleMentor — Sistema Especialista em Suplementação Esportiva

Assistente virtual especialista em suplementação esportiva e nutrição voltada a treinos, desenvolvido para a disciplina **Sistemas Especialistas para Web** (Pós-Graduação em Desenvolvimento Web — IFBA, Campus Vitória da Conquista).

O sistema integra quatro módulos principais:
- **Motor LLM (Google Gemini):** geração de respostas contextualizadas por persona, base de conhecimento e *guardrails* de domínio e segurança.
- **Cache Semântico (ChromaDB + Sentence-Transformers):** armazenamento vetorial local que recupera respostas para perguntas semanticamente equivalentes, reduzindo latência e consumo de tokens.
- **Servidor MCP (FastMCP):** motor de regras determinístico com 4 ferramentas especialistas (`consultar_suplemento`, `verificar_seguranca`, `calcular_dose_creatina` e `calcular_meta_proteina`).
- **Avaliação Automatizada (LLM-as-a-Judge + Testes Unitários):** validação contínua de precisão técnica, segurança clínica (*Red Teaming*) e aderência ao escopo.

## Estrutura do Projeto

```text
suplementor/
├── conhecimento/
│   ├── suplementos.json          # Base principal de conhecimento do domínio
│   ├── suplementos_menor.json    # Subconjunto de referência usado no julgamento
│   └── tabela_suplementos.json   # Ficha técnica e regras de contraindicação do servidor MCP
├── cache/                        # Banco vetorial local (criado automaticamente pelo ChromaDB)
├── .env.example                  # Modelo de variável de ambiente para a chave da API
├── app.py                        # Interface Web em Streamlit com telemetria em tempo real
├── cache.py                      # Geração de embeddings e controle do cache semântico
├── chat.py                       # Núcleo da persona, integração MCP e interface via terminal
├── ferramentas.py                # Servidor MCP com as 4 ferramentas especialistas
├── julgamento.py                 # Script de avaliação LLM-as-a-Judge (rubrica 0 a 5)
├── llm.py                        # Camada de comunicação com a API do Google Gemini
├── teste.py                      # Testes unitários de guardrail e domínio
├── relatorio_avaliacao.json      # Relatório gerado após a execução do julgamento.py
├── requirements.txt              # Dependências Python do projeto
└── README.md                     # Documentação de uso

Pré-requisitos e Instalação

# 1. Acesse a pasta do projeto e instale as dependências:

    cd suplementor
    pip install -r requirements.txt

# 2. Configure sua chave de API do Google Gemini:

    Obtenha uma chave em: https://aistudio.google.com/api-keys
    Copie o arquivo .env.example para .env e informe sua chave:
    GOOGLE_API_KEY=SUA_CHAVE_GOOGLE_AQUI

Como Executar

# 1. Subir o Servidor MCP (Terminal 1)

    python ferramentas.py

    *O serviço ficará escutando em http://0.0.0.0:8000/mcp.

# 2. Iniciar o Chat (Terminal 2)

# 2.1. Opção A — Interface Web (Streamlit):

    python -m streamlit run app.py

    *Acesse no navegador em http://localhost:8501.

# 2.2. Opção B — Interface de Linha de Comando (Terminal):

    python chat.py

    *Digite sair para encerrar a sessão.

Testes e Avaliação (LLM-as-a-Judge)

# Com o servidor ferramentas.py em execução no Terminal 1, execute no Terminal 2:

    # Testes unitários de domínio e guardrail
        python teste.py

    # Bateria completa LLM-as-a-Judge (alucinações, domínio e segurança clínica)
        python julgamento.py

    *Ao final do julgamento.py, as métricas consolidadas (taxa de aprovação, notas de 0 a 5 e tempo médio) são exibidas no terminal e gravadas em relatorio_avaliacao.json.

##### Aviso #####

As orientações fornecidas pelo SupleMentor possuem caráter estritamente informativo e educacional, não substituindo o acompanhamento de médicos ou nutricionistas.
