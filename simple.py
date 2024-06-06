import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def gerar_dados_simples(qtd_dados, desvio_padrao):
    np.random.seed(42)
    horas_estudo = np.random.rand(qtd_dados, 1) * 10  # Varia entre 0 e 10
    ruido = np.random.randn(qtd_dados, 1) * desvio_padrao  # Ruído com desvio padrão variável
    notas_alunos = 2 + 0.5 * horas_estudo + ruido  # Relação linear com ruído
    return horas_estudo, notas_alunos


def regressao_linear_simples(horas_estudo, notas_alunos):
    modelo = LinearRegression()
    modelo.fit(horas_estudo, notas_alunos)
    return modelo


def plotar_resultados_simples(horas_estudo, notas_alunos, modelo):
    notas_previstas = modelo.predict(horas_estudo)
    plt.figure(figsize=(8, 6))
    plt.scatter(horas_estudo, notas_alunos, color='blue', label='Dados Reais')
    plt.plot(horas_estudo, notas_previstas, color='red', linewidth=2, label='Linha de Regressão')
    plt.title('Regressão Linear Simples')
    plt.xlabel('Horas de Estudo')
    plt.ylabel('Notas dos Alunos')
    plt.legend()
    plt.grid(True)
    plt.show()


# Gerar dados com diferentes parâmetros
horas_estudo, notas_alunos = gerar_dados_simples(qtd_dados=50, desvio_padrao=1)

# Ajustar o modelo de regressão linear
modelo = regressao_linear_simples(horas_estudo, notas_alunos)

# Plotar os resultados
plotar_resultados_simples(horas_estudo, notas_alunos, modelo)
