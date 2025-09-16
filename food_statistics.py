class Statistics:

    def __init__(self, dataset):
        if not isinstance(dataset, dict):
            raise TypeError("O dataset deve ser um dicionário.")

        for _, values in dataset.items():
            if not isinstance(values, list):
                raise TypeError("Todos os valores no dicionário do dataset devem ser listas.")

        lengths = [len(values) for values in dataset.values()]
        if len(set(lengths)) != 1:
            raise ValueError("Todas as colunas no dataset devem ter o mesmo tamanho.")

        self.dataset = dataset

    def _get_column_data(self, column):
        if column not in self.dataset:
            raise KeyError(f"A coluna '{column}' não existe no dataset.")
        return self.dataset[column]

    def mean(self, column):
        valores = self._get_column_data(column)

        if not valores:
            return 0.0

        media_aritmetica = sum(valores) / len(valores)
        return float(media_aritmetica)

    def median(self, column):
        valores = self._get_column_data(column)

        if not valores:
            return 0.0

        valores_ordenados = sorted(valores)

        n = len(valores_ordenados)
        if n % 2 == 0:
            i = n // 2 - 1
            mediana = (valores_ordenados[i] + valores_ordenados[i + 1]) / 2
        else:
            i = n // 2
            mediana = valores_ordenados[i]

        return float(mediana)

    def mode(self, column):
        frequencias = self.absolute_frequency(column)

        if not frequencias:
            return []

        maior_frequencia = max(frequencias.values())

        moda = [item for item, freq in frequencias.items() if freq == maior_frequencia]

        return moda

    def stdev(self, column):
        variancia = self.variance(column)

        desvio_padrao = variancia ** 0.5

        return float(desvio_padrao)

    def variance(self, column):
        valores = self._get_column_data(column)

        if not valores:
            return 0.0

        media = self.mean(column)

        desvios_quadrados = ((item - media) ** 2 for item in valores)

        variancia = sum(desvios_quadrados) / len(valores)

        return float(variancia)

    def covariance(self, column_a, column_b):
        valores_a = self._get_column_data(column_a)
        valores_b = self._get_column_data(column_b)

        if len(valores_a) != len(valores_b) or len(valores_a) < 2:
            return 0.0

        media_a = self.mean(column_a)
        media_b = self.mean(column_b)

        soma_produto_desvios = sum((x - media_a) * (y - media_b) for x, y in zip(valores_a, valores_b))

        return float(soma_produto_desvios / len(valores_a))

    def itemset(self, column):
        valores = self._get_column_data(column)
        return set(valores)

    def absolute_frequency(self, column):
        valores = self._get_column_data(column)

        if not valores:
            return {}

        frequencias = {item: valores.count(item) for item in self.itemset(column)}

        return frequencias

    def relative_frequency(self, column):
        frequencias_absolutas = self.absolute_frequency(column)

        if not frequencias_absolutas:
            return {}

        total_amostras = len(self.dataset[column])

        frequencias_relativas = {item: freq / total_amostras for item, freq in frequencias_absolutas.items()}

        return frequencias_relativas

    def cumulative_frequency(self, column, frequency_method='absolute'):
        if frequency_method not in ['absolute', 'relative']:
            raise ValueError("O 'frequency_method' deve ser 'absolute' ou 'relative'.")

        valores = self._get_column_data(column)

        if not valores:
            return {}

        if frequency_method == 'absolute':
            frequencias = self.absolute_frequency(column)
        else:
            frequencias = self.relative_frequency(column)

        frequencia_acumulada = {}
        acumulado = 0

        for item in sorted(frequencias.keys()):
            acumulado += frequencias[item]
            frequencia_acumulada[item] = acumulado

        return frequencia_acumulada

    def conditional_probability(self, column, value1, value2):
        valores = self._get_column_data(column)

        if len(valores) < 2:
            return 0.0

        total_ocorrencias_value2 = valores.count(value2)

        sequencia = sum(1 for i in range(len(valores) - 1) if valores[i] == value2 and valores[i + 1] == value1)

        if total_ocorrencias_value2 > 0:
            return float(sequencia / total_ocorrencias_value2)

        return 0.0