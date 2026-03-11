# Transformer Encoder - Lab P1-02

Implementação da passagem direta (*forward pass*) de um bloco Encoder do Transformer utilizando Python, NumPy e Pandas.

Este laboratório foi desenvolvido na disciplina de **Tópicos em Inteligência Artificial** e tem como objetivo compreender o funcionamento interno da arquitetura proposta no artigo **Attention Is All You Need (Vaswani et al., 2017)**.

---

## Objetivo

Implementar do zero os principais componentes de um **Transformer Encoder**, incluindo:

- criação de um vocabulário simples
- geração de embeddings aleatórios
- mecanismo **Scaled Dot-Product Attention**
- **conexões residuais**
- **Layer Normalization**
- **Feed-Forward Network (FFN)**
- empilhamento de **6 camadas de encoder**

Ao final, o modelo recebe uma sequência de tokens e retorna uma representação vetorial contextualizada.

---

## Tecnologias utilizadas

- Python 3
- NumPy
- Pandas

---

## Como executar

### 1. Instalar dependências

```bash
pip install numpy pandas
