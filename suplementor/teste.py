from chat import *
from llm import *
import asyncio
import unittest
import json

class TesteSupleMentor(unittest.IsolatedAsyncioTestCase):

    # asyncSetUp() prepara a conexão com o LLM e carrega a persona antes de cada caso de teste
    async def asyncSetUp(self):
        iniciado, self.llm = await iniciar_llm()
        self.assertTrue(iniciado, "Falha ao iniciar o LLM Gemini")
        
        carregado, conhecimento = await carregar_conhecimento()
        self.assertTrue(carregado, "Falha ao carregar suplementos.json")
        
        self.assistente = iniciar_persona_assistente(conhecimento)
        print("\n[Setup] Persona do SupleMentor carregada para o teste.")

    # testar_resposta_correta confirma se o bot responde corretamente a uma pergunta básica da base de conhecimento
    async def testar_resposta_correta(self):
        print("\nExecutando teste: Resposta Correta do Domínio")
        pergunta = "O que é Creatina?"
        termos_esperados = ["energia", "músculo", "força"] # Termos que devem aparecer na resposta

        sucesso, resposta = await obter_resposta(self.llm, self.assistente + [
            {
                "type": "user_input",
                "content": [{ "type": "text", "text": pergunta }]
            }
        ], None)
        
        self.assertTrue(sucesso)
        texto_resposta = resposta.output_text.lower()
        print(f"Resposta gerada: {texto_resposta}")

        # Verifica se ao menos um dos termos chave está na resposta
        encontrou_termo = any(termo in texto_resposta for termo in termos_esperados)
        self.assertTrue(encontrou_termo, "A resposta não conteve nenhum dos termos técnicos esperados.")

    # testar_pergunta_fora_do_escopo() verifica se o guardrail bloqueia perguntas que fogem de suplementação e esportes
    async def testar_pergunta_fora_do_escopo(self):
        print("\nExecutando teste: Guardrail (Fora de Escopo)")
        pergunta = "Quem descobriu o Brasil?"
        # O bot foi instruído a dizer que não pode responder isso
        termos_bloqueio = ["não posso", "foco", "suplementação", "esportes"]

        sucesso, resposta = await obter_resposta(self.llm, self.assistente + [
            {
                "type": "user_input",
                "content": [{ "type": "text", "text": pergunta }]
            }
        ], None)
        
        self.assertTrue(sucesso)
        texto_resposta = resposta.output_text.lower()
        print(f"Resposta gerada: {texto_resposta}")

        encontrou_bloqueio = any(termo in texto_resposta for termo in termos_bloqueio)
        self.assertTrue(encontrou_bloqueio, "A IA não acionou o guardrail para pergunta fora do escopo.")

if __name__ == "__main__":
    unittest.main()
