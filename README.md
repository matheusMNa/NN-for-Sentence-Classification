# NN-for-Sentence-Classification
Esse repositório tem como objetivo utilizar otimização de hiperparâmetros de Redes Neurais (NN) para encarar um problema de classificação de texto, fazendo uso do dataset _PubMed RCT 20k_. O dado de entrada textual foi vetorizado por TF-IDF e processado pelos algoritmos de Aprendizado de Máquina (ML) SVC Linear e Regresão Logística para a construção de um baseline forte. Em seguida, uma Rede Neural (NN) do tipo Multi-Layer Perceptron (MLP) foi empregada para resolver o mesmo problema. Todos esses objetos -- vetorizador, modelos de ML e a MLP -- foram otimizados pelo framework _Optuna_. Todos os _scripts_ desse repositório foram desenvolvidos em Python. 

# !["Badge Ilum"](https://img.shields.io/badge/Ilum%20-%20purple) !["Badge Satus"](https://img.shields.io/badge/Status%20-%20Em_Desenvolvimento%20-%20orange)![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)


# 🛠️ Tecnologias utilizadas  🛠️
## IA
# ![Claude](https://img.shields.io/badge/Claude-D97757?style=for-the-badge&logo=claude&logoColor=white)
O Claude (Sonnet 4.5) foi utlizado para auxiliar no _debugging_ do software criado e na estruturação do pipeline de otimização no _Optuna_.

## Bibliotecas e módulos
As principais bibliotecas, módulos e pacotes utilizados foram:

- [Scikit-Learn](https://scikit-learn.org/stable/index.html)
- [Joblib](https://joblib.readthedocs.io/en/stable/)
- [Pytorch](https://docs.pytorch.org/docs/stable/index.html)
- [Optuna](https://optuna.readthedocs.io/en/stable/)
- [Numpy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [Conda](https://docs.conda.io/en/latest/)

Os códigos foram desenvolvidos em um ambiente virtual conda, cujas especificações estão contidas no arquivo [environment.yml](https://github.com/matheusMNa/Redes-Neurais/blob/main/environment.yml). A versão do Python utilizada é a 3.12.13.

# 💻 Instalação e Instruções 💻

Após realizar o download do repositório, é necessário criar um ambiente virtual que contenha, no mínimo, as principais bibliotecas. Recomenda-se que se faça uso de um ambiente virtual Conda. Após criar e ativar o ambiente virtual, abra a pasta `experiments` execute primeiro os arquivos de otimização e sequencialmente, os arquivos de construção final do modelo. Por fim, execute o arquivo de construção de gráficos no diretório `results` para obter a interpretação gráfica das métricas de desempenho, com foco no F1-Score. 

# 👥 Desenvolvedor do Projeto 👥


[<img loading="lazy" src="https://avatars.githubusercontent.com/u/88594280?v=4" width=115><br><sub>🐳Matheus Macedo do Nascimento🐳</sub>](https://github.com/matheusMNa)

Aluno do terceiro semestre de Ciência e Tecnologia, na Ilum Escola de Ciência

-Escreveu todo o código e estruturou este repositório no GitHub.


Agradecimento especial ao professor da matéria de Redes Neurais e Algoritmos Genéticos:

📍Professor Daniel Roberto Cassar



