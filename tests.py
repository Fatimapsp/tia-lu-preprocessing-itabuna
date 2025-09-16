import unittest
# Importa a classe a ser testada (assumindo que ela está no arquivo statistics.py)
from food_statistics import Statistics


class TestStatistics(unittest.TestCase):
    """
    Testes unitários para a classe Statistics.
    """

    def setUp(self):
        """
        Configura o ambiente de teste antes de cada método de teste.
        Cria uma base de dados com 5 colunas e 20 registros.
        """
        self.test_data = {
            'inteiros': [10, 8, 12, 8, 15, 6, 9, 10, 11, 14, 7, 13, 10, 16, 5, 10, 12, 9, 11, 14],
            'floats': [3.5, 2.1, 4.8, 2.1, 5.5, 1.2, 3.3, 4.0, 4.2, 5.0, 2.8, 4.9, 3.9, 5.8, 1.0, 3.5, 4.8, 3.3, 4.2,
                       5.0],
            'negativos': [-5, -2, 0, -2, 3, -10, -1, -5, 1, 2, -7, 3, -5, 6, -15, -5, 0, -1, 1, 2],
            'categorica': ['A', 'B', 'C', 'A', 'B', 'D', 'A', 'C', 'B', 'D', 'A', 'C', 'B', 'A', 'D', 'A', 'B', 'C',
                           'B', 'D'],
            'sequencial': [1, 2, 1, 3, 1, 2, 2, 3, 1, 3, 1, 2, 3, 2, 1, 2, 1, 3, 2, 1]
        }
        self.stats = Statistics(self.test_data)

    # ==================================================================
    # Testes de Casos de Sucesso
    # ==================================================================

    def test_mean(self):
        self.assertAlmostEqual(self.stats.mean('inteiros'), 10.5)
        self.assertAlmostEqual(self.stats.mean('floats'), 3.745)
        self.assertAlmostEqual(self.stats.mean('negativos'), -2.0)

    def test_median(self):
        # A lista ordenada tem 20 elementos. A mediana é a média dos elementos 10 e 11.
        # sorted = [5, 6, 7, 8, 8, 9, 9, 10, 10, 10, 10, 11, 11, 12, 12, 13, 14, 14, 15, 16]
        # (10 + 10) / 2 = 10.0
        self.assertAlmostEqual(self.stats.median('inteiros'), 10.0)
        # sorted = [-15, -10, -7, -5, -5, -5, -5, -2, -2, -1, -1, 0, 0, 1, 1, 2, 2, 3, 3, 6]
        # (-1 + -1) / 2 = -1.0
        self.assertAlmostEqual(self.stats.median('negativos'), -1.0)

    def test_mode(self):
        self.assertEqual(sorted(self.stats.mode('inteiros')), [10])
        # 'A' e 'B' aparecem 6 vezes cada
        self.assertEqual(sorted(self.stats.mode('categorica')), sorted(['A', 'B']))
        # '-5' aparece 4 vezes
        self.assertEqual(sorted(self.stats.mode('negativos')), [-5])

    def test_variance_and_stdev(self):
        # Usando um dataset simples para facilitar a verificação manual
        simple_data = {'col': [2, 4, 4, 4, 5, 5, 7, 9]}
        simple_stats = Statistics(simple_data)
        # Média = 5
        # Var = ((2-5)^2 + 3*(4-5)^2 + 2*(5-5)^2 + (7-5)^2 + (9-5)^2) / 8 = 32/8 = 4
        self.assertAlmostEqual(simple_stats.variance('col'), 4.0)
        self.assertAlmostEqual(simple_stats.stdev('col'), 2.0)

    def test_covariance(self):
        # Usando um dataset simples para facilitar a verificação
        cov_data = {'x': [1, 2, 3, 4], 'y': [4, 2, 3, 1]}
        cov_stats = Statistics(cov_data)
        # mean_x = 2.5, mean_y = 2.5
        # cov = ((-1.5*1.5) + (-0.5*-0.5) + (0.5*0.5) + (1.5*-1.5)) / 4
        # cov = (-2.25 + 0.25 + 0.25 - 2.25) / 4 = -4.0 / 4 = -1.0
        self.assertAlmostEqual(cov_stats.covariance('x', 'y'), -1.0)

    def test_itemset(self):
        self.assertEqual(self.stats.itemset('categorica'), {'A', 'B', 'C', 'D'})
        self.assertEqual(self.stats.itemset('inteiros'), {5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16})

    def test_absolute_frequency(self):
        expected = {'A': 6, 'B': 6, 'C': 4, 'D': 4}
        self.assertEqual(self.stats.absolute_frequency('categorica'), expected)

    def test_relative_frequency(self):
        expected = {'A': 0.3, 'B': 0.3, 'C': 0.2, 'D': 0.2}
        result = self.stats.relative_frequency('categorica')
        for key in expected:
            self.assertAlmostEqual(result[key], expected[key])

    def test_cumulative_frequency(self):
        # Teste com o método 'absolute'
        expected_abs = {'A': 6, 'B': 12, 'C': 16, 'D': 20}
        self.assertEqual(self.stats.cumulative_frequency('categorica', 'absolute'), expected_abs)

        # Teste com o método 'relative'
        expected_rel = {'A': 0.3, 'B': 0.6, 'C': 0.8, 'D': 1.0}
        result_rel = self.stats.cumulative_frequency('categorica', 'relative')
        for key in expected_rel:
            self.assertAlmostEqual(result_rel[key], expected_rel[key])

    def test_conditional_probability(self):
        # P(X=2 | X=1)
        # Contagem de '1': 8
        # Contagem da sequência (1, 2): 4
        # Probabilidade: 4 / 8 = 0.5
        self.assertAlmostEqual(self.stats.conditional_probability('sequencial', 2, 1), 0.5)

        # P(X=1 | X=3)
        # Contagem de '3': 5
        # Contagem da sequência (3, 1): 3
        # Probabilidade: 3 / 5 = 0.60
        self.assertAlmostEqual(self.stats.conditional_probability('sequencial', 1, 3), 0.60)

        # P(X=3 | X=2)
        # Contagem de '2': 7
        # Contagem da sequência (2, 3): 2
        # Probabilidade: 2 / 7 = 2/7
        self.assertAlmostEqual(self.stats.conditional_probability('sequencial', 3, 2), 2 / 7)

        # P(X=1 | X=4) -> '4' não existe, contagem do condicionante é 0
        self.assertEqual(self.stats.conditional_probability('sequencial', 1, 4), 0.0)

    # ==================================================================
    # Testes de Casos de Exceção
    # ==================================================================

    def test_init_exceptions(self):
        """Testa exceções durante a inicialização da classe."""
        # Caso 1: O dataset não é um dicionário
        with self.assertRaisesRegex(TypeError, "O dataset deve ser um dicionário."):
            Statistics([1, 2, 3])

        # Caso 2: Os valores do dicionário não são listas
        with self.assertRaisesRegex(TypeError, "Todos os valores no dicionário do dataset devem ser listas."):
            Statistics({'a': [1, 2], 'b': 'nao_e_lista'})

        # Caso 3: As listas têm tamanhos diferentes
        with self.assertRaisesRegex(ValueError, "Todas as colunas no dataset devem ter o mesmo tamanho."):
            Statistics({'a': [1, 2, 3], 'b': [4, 5]})

    def test_non_existent_column(self):
        """Testa se KeyError é levantado para uma coluna inexistente."""
        with self.assertRaisesRegex(KeyError, "A coluna 'coluna_falsa' não existe no dataset."):
            self.stats.mean('coluna_falsa')
        with self.assertRaises(KeyError):
            self.stats.median('coluna_que_nao_existe')

    def test_incompatible_data_type_for_numeric_methods(self):
        """Testa se um TypeError é levantado ao usar dados não-numéricos em cálculos."""
        with self.assertRaises(TypeError):
            self.stats.mean('categorica')
        with self.assertRaises(TypeError):
            self.stats.variance('categorica')
        with self.assertRaises(TypeError):
            # A mediana pode funcionar se o número de elementos for ímpar,
            # mas falhará se for par, pois tentará fazer uma operação matemática.
            # Vamos testar com uma lista par garantida de strings.
            str_stats = Statistics({'par': ['a', 'b', 'c', 'd']})
            str_stats.median('par')

    def test_empty_column_behavior(self):
        """Testa o comportamento dos métodos com uma coluna vazia."""
        empty_stats = Statistics({'vazia': [], 'outra': []})
        self.assertEqual(empty_stats.mean('vazia'), 0.0)
        self.assertEqual(empty_stats.median('vazia'), 0.0)
        self.assertEqual(empty_stats.mode('vazia'), [])
        self.assertEqual(empty_stats.variance('vazia'), 0.0)
        self.assertEqual(empty_stats.stdev('vazia'), 0.0)
        self.assertEqual(empty_stats.covariance('vazia', 'outra'), 0.0)
        self.assertEqual(empty_stats.itemset('vazia'), set())
        self.assertEqual(empty_stats.absolute_frequency('vazia'), {})
        self.assertEqual(empty_stats.relative_frequency('vazia'), {})
        self.assertEqual(empty_stats.cumulative_frequency('vazia'), {})

    def test_cumulative_frequency_invalid_method(self):
        """Testa a exceção para um método de frequência inválido."""
        with self.assertRaisesRegex(ValueError, "O 'frequency_method' deve ser 'absolute' ou 'relative'."):
            self.stats.cumulative_frequency('inteiros', frequency_method='metodo_invalido')


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)