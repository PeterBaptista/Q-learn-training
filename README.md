# 🎮 Q-Learning Game AI

Este projeto implementa um **agente de aprendizado por reforço** usando **Q-Learning** para aprender a navegar por um ambiente baseado em plataformas. O agente interage com o jogo por meio de uma conexão via socket e tenta maximizar sua pontuação ao longo dos episódios.

## 🚀 Recursos do projeto
✅ **Q-Learning** com tabela Q persistente para evitar perda de aprendizado.  
✅ **Exploração e Exploração** balanceadas com uma política ε-greedy ajustável.  
✅ **Decaimento da taxa de aprendizado (α) e exploração (ε)** para melhor convergência.  
✅ **Ciclos de treinamento configuráveis** para otimizar o aprendizado do agente.  
✅ **Critérios de recompensa ajustáveis**, incluindo penalizações e metas alcançadas.  

## ⚙ Como funciona?
1. O agente recebe um **estado do jogo** representado em formato binário.  
2. Ele escolhe uma ação (`left`, `right` ou `jump`) baseada na tabela Q.  
3. O ambiente retorna a **próxima observação e uma recompensa**.  
4. A tabela Q é **atualizada** de acordo com a equação do Q-Learning.  
5. O processo se repete por vários episódios até que o agente aprenda boas estratégias.  

## 🛠 Configuração e Execução
Para rodar o projeto, basta executar:
```bash
python client.py
```
O código perguntará a porta do servidor do jogo e carregará a **Q-table salva** (se disponível) para continuar o aprendizado.

## 📊 Parâmetros Importantes
- `LEARNING_RATE (α)`: Taxa de aprendizado ajustável entre 0.01 e 0.1.  
- `DISCOUNT_FACTOR (γ)`: Fator de desconto de **0.9222** para aprendizado a longo prazo.  
- `EPSILON (ε)`: Controla a exploração, diminuindo ao longo do tempo.  
- `MAX_STEPS`: Define o limite de ações por episódio para evitar loops infinitos.  
- `EPISODES_CYCLE`: Número de ciclos de aprendizado para garantir boa convergência.  

## 🏆 Objetivo do Agente
O agente busca alcançar a plataforma final **minimizando penalizações**, explorando e aprendendo a melhor sequência de ações.

📌 **Quer contribuir ou melhorar o agente?** Fique à vontade para enviar um **pull request** ou abrir uma **issue**! 🚀
