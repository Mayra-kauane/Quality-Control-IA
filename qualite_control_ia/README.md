# 📊 Qualite Control IA – Classificação de Defeitos em Chapas de Aço

Este projeto resolve um desafio de classificação multirrótulo para prever tipos de falhas em chapas de aço com base em variáveis de processo industriais.

---

## 🚀 Objetivo

Construir um modelo de Machine Learning capaz de prever múltiplos defeitos simultaneamente em uma mesma chapa.

---

## 🧠 Modelagem

- Tipo de problema: **Classificação Multirrótulo**
- Modelos testados:
  - Random Forest
  - Random Forest com `class_weight=balanced`
  - ✅ XGBoost (modelo final escolhido)
- Métrica principal: **F1-score por classe**

---

## ⚙️ Como executar

1. Clone este repositório:
```bash
git clone https://github.com/Mayra-kauane/Quality-Control-IA.git
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Execute o pipeline:
```bash
python main.py
```

---

## 📈 Resultados

Gráficos gerados:
- Frequência por tipo de falha
- F1-score por classe (XGBoost)
- Comparação entre modelos (RF vs XGBoost)

Arquivo de submissão gerado:
```
data/final_submission.xlsx
```

---

## ✅ Conclusão

O modelo XGBoost mostrou desempenho superior em classes desbalanceadas. A estrutura modular do projeto facilita reuso, testes futuros e integração com APIs.

---
