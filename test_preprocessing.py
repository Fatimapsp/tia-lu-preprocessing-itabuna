import unittest
from unittest.mock import patch
import copy

# Importa as classes do seu arquivo
from preprocessing import Preprocessing, MissingValueProcessor, Scaler, Encoder

class TestMissingValueProcessor(unittest.TestCase):
    
    def setUp(self):
        """Cria um dataset fresco para cada teste."""
        self.data = {
            'idade': [20, 30, None, 50],
            'salario': [500, None, 800, 1200],
            'cidade': ['A', 'B', 'C', None]
        }
    
    def test_isna(self):
        processor = MissingValueProcessor(copy.deepcopy(self.data))
        # Testa a busca por nulos em uma coluna específica
        result = processor.isna(columns={'idade'})
        self.assertEqual(result['idade'], [None])
        self.assertEqual(len(result['idade']), 1)
        
        # Testa a busca por nulos em todas as colunas (comportamento padrão)
        result_all_cols = processor.isna()
        self.assertEqual(len(result_all_cols['idade']), 4) # Todas as linhas têm algum nulo

    def test_notna(self):
        processor = MissingValueProcessor(copy.deepcopy(self.data))
        # Testa a busca por não nulos em colunas específicas
        result = processor.notna(columns={'idade', 'salario'})
        self.assertEqual(result['idade'], [20, 50])
        self.assertEqual(len(result['idade']), 2)

    def test_fillna_mean(self):
        processor = MissingValueProcessor(copy.deepcopy(self.data))
        processor.fillna(columns={'idade'}, method='mean')
        # Média de (20+30+50)/3 = 33.333...
        self.assertAlmostEqual(processor.dataset['idade'][2], 33.3333333)

    def test_fillna_mode(self):
        data_with_mode = {'cat': ['A', 'B', 'A', None]}
        processor = MissingValueProcessor(data_with_mode)
        processor.fillna(columns={'cat'}, method='mode')
        self.assertEqual(processor.dataset['cat'][3], 'A')

    def test_dropna(self):
        processor = MissingValueProcessor(copy.deepcopy(self.data))
        processor.dropna(columns={'cidade'})
        self.assertEqual(len(processor.dataset['cidade']), 3)
        self.assertNotIn(None, processor.dataset['cidade'])


class TestScaler(unittest.TestCase):

    def setUp(self):
        self.data = {'feature': [10, 20, 30, 40, 50]}

    def test_minMax_scaler(self):
        scaler = Scaler(copy.deepcopy(self.data))
        scaler.minMax_scaler(columns={'feature'})
        expected = [0.0, 0.25, 0.5, 0.75, 1.0]
        for original, scaled in zip(expected, scaler.dataset['feature']):
            self.assertAlmostEqual(original, scaled)

    def test_standard_scaler(self):
        scaler = Scaler(copy.deepcopy(self.data))
        scaler.standard_scaler(columns={'feature'})
        # Mean=30, StdDev=sqrt( ((-20)^2 + (-10)^2 + 0^2 + 10^2 + 20^2) / 5 ) = sqrt(1000/5) = sqrt(200) ~= 14.142
        expected = [-1.4142, -0.7071, 0.0, 0.7071, 1.4142]
        for original, scaled in zip(expected, scaler.dataset['feature']):
            self.assertAlmostEqual(original, scaled, places=4)


class TestEncoder(unittest.TestCase):
    
    def setUp(self):
        self.data = {'cor': ['azul', 'verde', 'vermelho', 'azul']}

    def test_label_encode(self):
        encoder = Encoder(copy.deepcopy(self.data))
        encoder.label_encode(columns={'cor'})
        # 'azul':0, 'verde':1, 'vermelho':2 (ordem alfabética)
        expected = [0, 1, 2, 0]
        self.assertEqual(encoder.dataset['cor'], expected)

    def test_oneHot_encode(self):
        encoder = Encoder(copy.deepcopy(self.data))
        encoder.oneHot_encode(columns={'cor'})
        self.assertIn('cor_azul', encoder.dataset)
        self.assertIn('cor_verde', encoder.dataset)
        self.assertIn('cor_vermelho', encoder.dataset)
        self.assertNotIn('cor', encoder.dataset)
        self.assertEqual(encoder.dataset['cor_azul'], [1, 0, 0, 1])
        self.assertEqual(encoder.dataset['cor_verde'], [0, 1, 0, 0])


class TestPreprocessingFacade(unittest.TestCase):

    def setUp(self):
        self.data = {'a': [1, 2], 'b': [3, 4]}

    def test_validation_fails_with_mismatched_lengths(self):
        """Garante que o construtor levanta um erro para colunas de tamanhos diferentes."""
        invalid_data = {'a': [1, 2], 'b': [3]}
        with self.assertRaises(ValueError):
            Preprocessing(invalid_data)

    # Patch indica o local ONDE o objeto é usado, não onde ele é definido.
    @patch('preprocessing_lib.MissingValueProcessor')
    @patch('preprocessing_lib.Scaler')
    @patch('preprocessing_lib.Encoder')
    @patch('preprocessing_lib.Statistics')
    def test_facade_methods_call_correct_implementations(self, MockStats, MockEncoder, MockScaler, MockMVP):
        """
        Testa se os métodos de fachada da classe Preprocessing chamam
        corretamente os métodos das classes internas (usando mocks).
        """
        # Instancia mocks para as implementações
        mock_mvp_instance = MockMVP.return_value
        mock_scaler_instance = MockScaler.return_value
        mock_encoder_instance = MockEncoder.return_value

        # Cria a instância da classe principal
        preprocessor = Preprocessing(self.data)

        # Testa os atalhos de MissingValueProcessor
        preprocessor.dropna(columns={'a'})
        mock_mvp_instance.dropna.assert_called_once_with(columns={'a'})

        preprocessor.fillna(columns={'b'}, method='median')
        mock_mvp_instance.fillna.assert_called_once_with(columns={'b'}, method='median', default_value=0)
        
        preprocessor.isna(columns={'a'})
        mock_mvp_instance.isna.assert_called_once_with(columns={'a'})
        
        # Testa o atalho de Scaler
        preprocessor.scale(columns={'a'}, method='standard')
        mock_scaler_instance.standard_scaler.assert_called_once_with(columns={'a'})
        mock_scaler_instance.minMax_scaler.assert_not_called()

        preprocessor.scale(columns={'b'}, method='minMax')
        mock_scaler_instance.minMax_scaler.assert_called_once_with(columns={'b'})

        # Testa o atalho de Encoder
        preprocessor.encode(columns={'b'}, method='oneHot')
        mock_encoder_instance.oneHot_encode.assert_called_once_with(columns={'b'})
        mock_encoder_instance.label_encode.assert_not_called()
        
    def test_scale_raises_error_for_invalid_method(self):
        preprocessor = Preprocessing(self.data)
        with self.assertRaises(ValueError):
            preprocessor.scale(method='invalid_method')
            
if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)