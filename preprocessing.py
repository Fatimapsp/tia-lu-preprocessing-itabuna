from food_statistics import Statistics
from typing import Dict, List, Set, Any

class MissingValueProcessor:
    """
    Processa valores ausentes (representados como None) no dataset.
    """
    def __init__(self, dataset: Dict[str, List[Any]]):
        self.dataset = dataset

    def _get_target_columns(self, columns: Set[str]) -> List[str]:
        """Retorna as colunas a serem processadas. Se 'columns' for vazio, retorna todas as colunas."""
        return list(columns) if columns else list(self.dataset.keys())

    def isna(self, columns: Set[str] = None) -> Dict[str, List[Any]]:
        target_columns = self._get_target_columns(columns)

        if not target_columns or not self.dataset:
            return {}

        num_rows = len(self.dataset[target_columns[0]])
        result_dataset = {col: [] for col in self.dataset.keys()}

        for i in range(num_rows):
            if any(self.dataset[col][i] is None for col in target_columns):
                for col in self.dataset.keys():
                    result_dataset[col].append(self.dataset[col][i])

        return result_dataset

    def notna(self, columns: Set[str] = None) -> Dict[str, List[Any]]:
        target_columns = self._get_target_columns(columns)

        if not target_columns or not self.dataset:
            return {}

        num_rows = len(self.dataset[target_columns[0]])
        result_dataset = {col: [] for col in self.dataset.keys()}

        for i in range(num_rows):
            if all(self.dataset[col][i] is not None for col in target_columns):
                for col in self.dataset.keys():
                    result_dataset[col].append(self.dataset[col][i])

        return result_dataset

    def fillna(self, columns: Set[str] = None, method: str = 'mean', default_value: Any = 0):
        target_columns = self._get_target_columns(columns)
        if not target_columns or not self.dataset:
            return

        for col in target_columns:
            fill_value = None
            if method == 'mean':
                clean_values = [v for v in self.dataset[col] if v is not None]
                if clean_values:
                    stats_clean = Statistics({col: clean_values})
                    fill_value = stats_clean.mean(col)
                else:
                    fill_value = default_value
            elif method == 'median':
                clean_values = [v for v in self.dataset[col] if v is not None]
                if clean_values:
                    stats_clean = Statistics({col: clean_values})
                    fill_value = stats_clean.median(col)
                else:
                    fill_value = default_value
            elif method == 'mode':
                clean_values = [v for v in self.dataset[col] if v is not None]
                if clean_values:
                    stats_clean = Statistics({col: clean_values})
                    mode_result = stats_clean.mode(col)
                    if mode_result:
                        fill_value = mode_result[0]
                    else:
                        fill_value = default_value
                else:
                    fill_value = default_value
            elif method == 'default_value':
                fill_value = default_value
            else:
                raise ValueError(f"Método '{method}' não suportado.")

            for i in range(len(self.dataset[col])):
                if self.dataset[col][i] is None:
                    self.dataset[col][i] = fill_value

    def dropna(self, columns: Set[str] = None):
        target_columns = self._get_target_columns(columns)

        if not target_columns or not self.dataset:
            return

        valid_rows_indices = []
        num_rows = len(self.dataset[target_columns[0]])

        for i in range(num_rows):
            is_valid = True
            for col in target_columns:
                if self.dataset[col][i] is None:
                    is_valid = False
                    break
            if is_valid:
                valid_rows_indices.append(i)

        new_dataset = {col: [] for col in self.dataset.keys()}
        for i in valid_rows_indices:
            for col in self.dataset.keys():
                new_dataset[col].append(self.dataset[col][i])

        self.dataset.clear()
        self.dataset.update(new_dataset)


class Scaler:
    """
    Aplica transformações de escala em colunas numéricas do dataset.
    """
    def __init__(self, dataset: Dict[str, List[Any]]):
        self.dataset = dataset

    def _get_target_columns(self, columns: Set[str]) -> List[str]:
        return list(columns) if columns else list(self.dataset.keys())

    def minMax_scaler(self, columns: Set[str] = None):
        """
        Aplica a normalização Min-Max ($X_{norm} = \frac{X - X_{min}}{X_{max} - X_{min}}$)
        nas colunas especificadas. Modifica o dataset.

        Args:
            columns (Set[str]): Colunas para aplicar o scaler. Se vazio, tenta aplicar a todas.
        """
        pass

    def standard_scaler(self, columns: Set[str] = None):
        """
        Aplica a padronização Z-score ($X_{std} = \frac{X - \mu}{\sigma}$)
        nas colunas especificadas. Modifica o dataset.

        Args:
            columns (Set[str]): Colunas para aplicar o scaler. Se vazio, tenta aplicar a todas.
        """
        pass

class Encoder:
    """
    Aplica codificação em colunas categóricas.
    """
    def __init__(self, dataset: Dict[str, List[Any]]):
        self.dataset = dataset

    def label_encode(self, columns: Set[str]):
        """
        Converte cada categoria em uma coluna em um número inteiro.
        Modifica o dataset.

        Args:
            columns (Set[str]): Colunas categóricas para codificar.
        """
        pass

    def oneHot_encode(self, columns: Set[str]):
        """
        Cria novas colunas binárias para cada categoria nas colunas especificadas (One-Hot Encoding).
        Modifica o dataset adicionando e removendo colunas.

        Args:
            columns (Set[str]): Colunas categóricas para codificar.
        """
        pass


class Preprocessing:
    """
    Classe principal que orquestra as operações de pré-processamento de dados.
    """
    def __init__(self, dataset: Dict[str, List[Any]]):
        self.dataset = dataset
        self._validate_dataset_shape()
        
        # Atributos compostos para cada tipo de tarefa
        self.statistics = Statistics(self.dataset)
        self.missing_values = MissingValueProcessor(self.dataset)
        self.scaler = Scaler(self.dataset)
        self.encoder = Encoder(self.dataset)

    def _validate_dataset_shape(self):
        """
        Valida se todas as listas (colunas) no dicionário do dataset
        têm o mesmo comprimento.
        """
        pass

    def isna(self, columns: Set[str] = None) -> Dict[str, List[Any]]:
        """
        Atalho para missing_values.isna(). Retorna as linhas com valores nulos.
        """
        return self.missing_values.isna(columns)

    def notna(self, columns: Set[str] = None) -> Dict[str, List[Any]]:
        """
        Atalho para missing_values.notna(). Retorna as linhas sem valores nulos.
        """
        return self.missing_values.notna(columns)

    def fillna(self, columns: Set[str] = None, method: str = 'mean', default_value: Any = 0):
        """
        Atalho para missing_values.fillna(). Preenche valores nulos.
        Retorna 'self' para permitir encadeamento de métodos.
        """
        self.missing_values.fillna(columns, method, default_value)
        return self

    def dropna(self, columns: Set[str] = None):
        """
        Atalho para missing_values.dropna(). Remove linhas com valores nulos.
        Retorna 'self' para permitir encadeamento de métodos.
        """
        self.missing_values.dropna(columns)
        return self

    def scale(self, columns: Set[str] = None, method: str = 'minMax'):
        """
        Aplica escalonamento nas colunas especificadas.

        Args:
            columns (Set[str]): Colunas para aplicar o escalonamento.
            method (str): O método a ser usado: 'minMax' ou 'standard'.

        Retorna 'self' para permitir encadeamento de métodos.
        """
        if method == 'minMax':
            self.scaler.minMax_scaler(columns)
        elif method == 'standard':
            self.scaler.standard_scaler(columns)
        else:
            raise ValueError(f"Método de escalonamento '{method}' não suportado. Use 'minMax' ou 'standard'.")
        return self

    def encode(self, columns: Set[str], method: str = 'label'):
        """
        Aplica codificação nas colunas especificadas.

        Args:
            columns (Set[str]): Colunas para aplicar a codificação.
            method (str): O método a ser usado: 'label' ou 'oneHot'.
        
        Retorna 'self' para permitir encadeamento de métodos.
        """
        if not columns:
            print("Aviso: Nenhuma coluna especificada para codificação. Nenhuma ação foi tomada.")
            return self

        if method == 'label':
            self.encoder.label_encode(columns)
        elif method == 'oneHot':
            self.encoder.oneHot_encode(columns)
        else:
            raise ValueError(f"Método de codificação '{method}' não suportado. Use 'label' ou 'oneHot'.")
        return self
