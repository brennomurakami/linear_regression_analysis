import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression


def gerar_dados(qtd_dados, desvio_padrao):
    np.random.seed(42)
    horas_estudo = np.random.rand(qtd_dados, 1) * 10  # Varia entre 0 e 10
    horas_sono = np.random.rand(qtd_dados, 1) * 10  # Varia entre 0 e 10
    ruido = np.random.randn(qtd_dados, 1) * desvio_padrao  # Ruído com desvio padrão variável
    notas_alunos = 2 + 0.5 * horas_estudo + 1.5 * horas_sono + ruido  # Relação linear com ruído
    return horas_estudo, horas_sono, notas_alunos


def regressao_linear(horas_estudo, horas_sono, notas_alunos):
    X = np.concatenate((horas_estudo, horas_sono), axis=1)
    modelo = LinearRegression()
    modelo.fit(X, notas_alunos)
    return modelo


def plotar_resultados(horas_estudo, horas_sono, notas_alunos, modelo):
    fig = plt.figure(figsize=(12, 6))

    # Gráfico 3D para visualizar as duas variáveis independentes e a variável dependente
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(horas_estudo, horas_sono, notas_alunos, color='blue', label='Dados Reais')
    ax.set_xlabel('Horas de Estudo')
    ax.set_ylabel('Horas de Sono')
    ax.set_zlabel('Notas dos Alunos')
    ax.set_title('Regressão Linear Múltipla - Dados Reais')

    # Plano de regressão
    xx, yy = np.meshgrid(np.linspace(0, 10, 10), np.linspace(0, 10, 10))
    zz = modelo.intercept_ + modelo.coef_[0][0] * xx + modelo.coef_[0][1] * yy
    ax.plot_surface(xx, yy, zz, alpha=0.5, color='red')

    # Gráfico 2D para visualizar as previsões
    ax = fig.add_subplot(122)
    notas_previstas = modelo.predict(np.concatenate((horas_estudo, horas_sono), axis=1))
    ax.scatter(notas_alunos, notas_previstas, color='blue', label='Previsões')
    ax.plot([notas_alunos.min(), notas_alunos.max()], [notas_alunos.min(), notas_alunos.max()], 'k--', lw=2)
    ax.set_xlabel('Notas Reais')
    ax.set_ylabel('Notas Previstas')
    ax.set_title('Regressão Linear Múltipla - Previsões')

    plt.tight_layout()
    plt.show()


# Gerar dados com diferentes parâmetros
horas_estudo, horas_sono, notas_alunos = gerar_dados(qtd_dados=50, desvio_padrao=2)

# Ajustar o modelo de regressão linear
modelo = regressao_linear(horas_estudo, horas_sono, notas_alunos)

# Plotar os resultados
plotar_resultados(horas_estudo, horas_sono, notas_alunos, modelo)