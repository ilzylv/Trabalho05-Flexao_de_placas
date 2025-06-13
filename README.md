# Análise de Placas Retangulares com Séries de Navier
## Descrição
Este projeto consiste em um programa em Python que utiliza matemática simbólica e numérica para realizar a análise de placas retangulares finas com bordas simplesmente apoiadas, baseando-se na Teoria de Placas de Kirchhoff-Love. A solução para o campo de deflexões é obtida através do método de Navier, que emprega séries duplas de Fourier. O script foi desenvolvido como uma ferramenta de estudo para implementar as soluções apresentadas nos exemplos do livro "Plates and Shells: Theory and Analysis" de A. C. Ugural (2018).

## Exemplos Implementados (Ugural, 2018)
O script resolve e apresenta os resultados para os seguintes casos:

- Exemplo 5.1: Placa sob carga uniformemente distribuída (p(x,y) = p₀).

- Exemplo 5.2: Placa sob carga senoidal (p(x,y) = p₀*sin(πx/a)*sin(πy/b)).

- Exemplo 5.3: Placa sob carga concentrada no centro (P).

## Instalação
1. Certifique-se de ter as seguintes bibliotecas Python instaladas:
    ```bash
    pip install sympy numpy matplotlib
  
2. Basta executar o script diretamente no terminal. Ele irá processar e exibir os resultados para os três exemplos em sequência, gerando todos os gráficos.
    ```bash
    python nome_do_seu_arquivo.py

## Referência
Ugural, A. C. (2018). Plates and Shells: Theory and Analysis. CRC press.
