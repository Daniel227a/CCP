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

Ao longo da condução da simulação, observou-se um aumento progressivo na janela, coincidindo com um incremento nas recompensas, foram identificadas taxas vantajosas de vazão de rede mediante a implementação do modelo Flat. Este achado ressalta a eficácia do referido modelo na otimização do desempenho da rede, resultando em taxas de transmissão de dados que se mostraram positivas durante a simulação.

A escolha do modelo Flat  permitiu a análise específica de seu impacto nas taxas de vazão, evidenciando suas contribuições no contexto do experimento como pode ser visto na Figura.
<a target="_blank" align="center">
  <img align="center"  height="600" width="1000"  src="https://github.com/Daniel227a/CCP/blob/main/graficos/flat31.png">
</a>
