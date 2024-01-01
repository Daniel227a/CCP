# Otimização do Desempenho do Protocolo TCP com CCP

A otimização do desempenho do protocolo de transporte confiável da Internet, conhecido como TCP, implica na concepção de uma solução adaptativa que determine o tamanho ideal da janela de congestionamento em uma conexão TCP. Dessa forma, ao perceber as condições do ambiente, o algoritmo de CCP buscará maximizar a vazão da rede de comunicação, apresentando uma abordagem alternativa ao funcionamento convencional do TCP.
# Objetivos Específicos

Para realizar os objetivos principais do projeto é necessário tratar com os seguintes micro-objetivos.

- **Elaboração de um algoritmo utilizando técnicas de QL** que consiga reconhecer e modificar o ambiente gerado pela rede.
- **Construção de um ambiente de simulação em NS3** visando garantir comparações fidedignas entre os algoritmos em um ambiente de rede real.
- **Aplicação de métodos de estatística descritiva** utilizadas para o comparativo entre os experimentos.
- **Avaliação de Desempenho por análise de Variância** onde se analisa quais fatores e níveis mais influenciam na variável de reposta, tempo de convergência e vazão em redes de comunicação.
- # Experimentos

Com o propósito de desenvolver um algoritmo destinado ao aprimoramento da vazão em redes de comunicação, a análise dos experimentos concentrou-se na avaliação dos efeitos que diferentes configurações de redes neurais e indução de taxas de erro no canal de comunicação.

Além disso, foram examinadas as influências de fatores específicos, tais como a taxa de erro em pacotes, assim como o impacto do algoritmo de *Q-Learning*. Adicionalmente, a influência do *learning rate* também foi objeto de análise.

Este estudo buscou não apenas compreender a eficácia das redes neurais na otimização da vazão, mas também discernir as contribuições individuais de fatores críticos, como a taxa de erro em pacotes e os parâmetros fundamentais do algoritmo de *Q-Learning*, como o *learning rate*, no contexto da melhoria da eficiência de comunicação em redes.

Os resultados obtidos neste estudo foram derivados a partir da realização de 120 experimentos conduzidos no ambiente de simulação de rede NS3.

Durante a simulação, cada rede neural apresentou um comportamento diferente ao atribuir a janela de congestionamento cwnd com base em suas decisões. O algoritmo obteve as chamadas recompensas por decisões corretas.Três modelos distintos foram implementados para realizar esta análise. O primeiro modelo, denominado  Flat, foi configurado com apenas duas camadas ocultas.
# Flat Neural Network (Flat)
Ao longo da condução da simulação, observou-se um aumento progressivo na janela, coincidindo com um incremento nas recompensas, foram identificadas taxas vantajosas de vazão de rede mediante a implementação do modelo Flat. Este achado ressalta a eficácia do referido modelo na otimização do desempenho da rede, resultando em taxas de transmissão de dados que se mostraram positivas durante a simulação.

A escolha do modelo Flat  permitiu a análise específica de seu impacto nas taxas de vazão, evidenciando suas contribuições no contexto do experimento como pode ser visto na Figura.
<a target="_blank" align="center">
  <img align="center"  height="600" width="1000"  src="https://github.com/Daniel227a/CCP/blob/main/graficos/flat31.png">
</a>
# Neural Network  (NN)

O segundo modelo, intitulado *Neural Network (NN)*, apresentava uma arquitetura mais complexa, composta por quatro camadas ocultas.

Nesta configuração específica, constatou-se uma média inferior nas taxas de vazão. Essa observação aponta para a influência da complexidade arquitetônica na eficiência da rede, destacando a importância de considerações detalhadas sobre a estrutura do modelo na análise do desempenho das taxas de transmissão de dados, conforme ilustrado na Figura.
<a target="_blank" align="center">
  <img align="center"  height="600" width="1000"  src="https://github.com/Daniel227a/CCP/blob/main/graficos/DM-_2_-_1_.jpg">
</a>
#  Deep Q-Learning Model (DMP)

Por fim, o terceiro modelo *Deep Q-Learning Model (DMP)* possuía uma estrutura ainda mais elaborada, compreendendo oito camadas ocultas. Estas configurações variadas tinham como finalidade investigar a influência dos diferentes algoritmos de *Q-Learning* no desempenho da rede.

No transcorrer da simulação, observou-se que o modelo *DMP* manifestou uma vazão de rede inferior em comparação com os demais modelos neurais, como pode ser visto na Figura.
<a target="_blank" align="center">
  <img align="center"  height="600" width="1000"  src="https://github.com/Daniel227a/CCP/blob/main/graficos/DPM-_2_-_1_.png">
</a>
# Influência do Algoritmo Q-Learning e da Taxa de Aprendizado na Vazão da Rede

A Figura  apresenta o desempenho do algoritmo *Q-Learning* para três diferentes arquiteturas: *Flat*, *NN* e *Deep DMP*. As colunas representam a Vazão da Rede máxima alcançada pelo algoritmo em função da taxa de aprendizado. Como pode ser observado, o algoritmo *Flat* apresenta o melhor desempenho, seguido pelo *NN* e pelo *Deep DMP*. O *DMP* é o algoritmo que apresenta o pior desempenho, com a Vazão da Rede máxima significativamente inferior aos demais algoritmos.

<a target="_blank" align="center">
  <img align="center"  height="600" width="1000"  src="https://github.com/Daniel227a/CCP/blob/main/graficos/Chart%20of%20Max.%20Throughput%2C%20learning_rate%2C%20Q-Learning%20Algorithm.png">
</a>
Este resultado pode ser explicado pelo fato de que o *DMP* é um algoritmo mais complexo do que os demais. Ele possui uma estrutura de rede neural mais profunda, o que requer mais recursos computacionais para ser executado.

Os resultados deste estudo sugerem que o modelo *Flat* é a arquitetura mais adequada para o algoritmo *Q-Learning* em ambientes complexos. O *NN* pode ser uma alternativa viável.

# Análise do Impacto do Algoritmo Q-Learning no Tempo de Convergência

A Figura  apresenta o tempo de convergência do algoritmo *Q-Learning* para três diferentes arquiteturas: *Flat*, *NN* e *DMP*. As curvas representam o número de etapas necessárias para que o algoritmo atinja a convergência em função da taxa de aprendizado. Como pode ser observado, o algoritmo *Flat* apresenta o melhor desempenho, com um tempo de convergência significativamente inferior aos demais algoritmos. O *DMP* é o algoritmo que apresenta o pior desempenho, com um tempo de convergência mais de 10 vezes maior que o algoritmo *Flat*.

<a target="_blank" align="center">
  <img align="center"  height="600" width="1000"  src="https://github.com/Daniel227a/CCP/blob/main/graficos/Line%20Plot%20of%20Convergence%20Step.png">
</a>
