from food_statistics import Statistics
from typing import Dict, List, Set, Any


class MissingValueProcessor:
    """Processa valores ausentes (representados como None) no dataset."""
    def __init__(self, dataset: Dict[str, List[Any]]):
        self.dataset = dataset

    def _get_target_columns(self, columns: Set[str]) -> List[str]:
        """Retorna as colunas a serem processadas. Se 'columns' for vazio ou None, retorna todas as colunas."""
        if columns is None or not columns:
            return list(self.dataset.keys())
        return list(columns)

    def isna(self, columns: Set[str] = None) -> Dict[str, List[Any]]:
        """Retorna as linhas que contêm valores ausentes nas colunas especificadas."""
        target_columns = self._get_target_columns(columns)

        if not target_columns or not self.dataset:
            return {col: [] for col in self.dataset.keys()} if self.dataset else {}

        num_rows = len(self.dataset[target_columns[0]])
        result_dataset = {col: [] for col in self.dataset.keys()}

        for row_index in range(num_rows):
            has_missing_value = any(self.dataset[col][row_index] is None for col in target_columns)
            if has_missing_value:
                for col in self.dataset.keys():
                    result_dataset[col].append(self.dataset[col][row_index])

        return result_dataset

    def notna(self, columns: Set[str] = None) -> Dict[str, List[Any]]:
        """Retorna as linhas que NÃO contêm valores ausentes nas colunas especificadas."""
        target_columns = self._get_target_columns(columns)

        if not target_columns or not self.dataset:
            return {col: [] for col in self.dataset.keys()} if self.dataset else {}

        num_rows = len(self.dataset[target_columns[0]])
        result_dataset = {col: [] for col in self.dataset.keys()}

        for row_index in range(num_rows):
            has_all_values = all(self.dataset[col][row_index] is not None for col in target_columns)
            if has_all_values:
                for col in self.dataset.keys():
                    result_dataset[col].append(self.dataset[col][row_index])

        return result_dataset

    def fillna(self, columns: Set[str] = None, method: str = 'mean', default_value: Any = 0):
        """Preenche valores ausentes usando diferentes estratégias estatísticas."""
        target_columns = self._get_target_columns(columns)
        if not target_columns or not self.dataset:
            return

        for column_name in target_columns:
            if column_name not in self.dataset:
                continue

            fill_value = None
            if method == 'mean':
                numeric_values = [value for value in self.dataset[column_name]
                                  if value is not None and isinstance(value, (int, float))]
                if numeric_values:
                    statistics_calculator = Statistics({column_name: numeric_values})
                    fill_value = statistics_calculator.mean(column_name)
                else:
                    fill_value = default_value

            elif method == 'median':
                numeric_values = [value for value in self.dataset[column_name]
                                  if value is not None and isinstance(value, (int, float))]
                if numeric_values:
                    statistics_calculator = Statistics({column_name: numeric_values})
                    fill_value = statistics_calculator.median(column_name)
                else:
                    fill_value = default_value

            elif method == 'mode':
                valid_values = [value for value in self.dataset[column_name] if value is not None]
                if valid_values:
                    statistics_calculator = Statistics({column_name: valid_values})
                    mode_results = statistics_calculator.mode(column_name)
                    if mode_results:
                        fill_value = mode_results[0]
                    else:
                        fill_value = default_value
                else:
                    fill_value = default_value

            elif method == 'default_value':
                fill_value = default_value
            else:
                raise ValueError(f"Método '{method}' não suportado. Use: 'mean', 'median', 'mode', 'default_value'")

            for position_index in range(len(self.dataset[column_name])):
                if self.dataset[column_name][position_index] is None:
                    self.dataset[column_name][position_index] = fill_value

    def dropna(self, columns: Set[str] = None):
        """Remove linhas que contêm valores ausentes nas colunas especificadas."""
        target_columns = self._get_target_columns(columns)

        if not target_columns or not self.dataset:
            return

        valid_row_indices = []
        num_rows = len(self.dataset[target_columns[0]]) if target_columns else 0

        for row_index in range(num_rows):
            is_row_valid = True
            for column_name in target_columns:
                if self.dataset[column_name][row_index] is None:
                    is_row_valid = False
                    break
            if is_row_valid:
                valid_row_indices.append(row_index)

        new_dataset = {col: [] for col in self.dataset.keys()}
        for valid_index in valid_row_indices:
            for column_name in self.dataset.keys():
                new_dataset[column_name].append(self.dataset[column_name][valid_index])

        self.dataset.clear()
        self.dataset.update(new_dataset)

class Scaler:
    """Aplica transformações de escala em colunas numéricas do dataset."""
    def __init__(self, dataset: Dict[str, List[Any]]):
        self.dataset = dataset

    def _get_target_columns(self, columns: Set[str]) -> List[str]:
        """Retorna colunas alvo ou todas se não especificado."""
        if columns is None or not columns:
            return list(self.dataset.keys())
        return list(columns)

    def _validate_numeric_data(self, column_name: str) -> bool:
        """Valida se a coluna contém apenas dados numéricos."""
        if column_name not in self.dataset:
            return False

        column_data = self.dataset[column_name]
        return all(isinstance(value, (int, float)) and not isinstance(value, bool)
                   for value in column_data if value is not None)

    def minMax_scaler(self, columns: Set[str] = None):
        """Aplica a normalização Min-Max ($X_{norm} = \frac{X - X_{min}}{X_{max} - X_{min}}$)
        nas colunas especificadas. Modifica o dataset. """
        target_columns = self._get_target_columns(columns)
        for column_name in target_columns:
            if not self._validate_numeric_data(column_name):
                continue

            column_data = self.dataset[column_name]
            if not column_data:
                continue

            min_value = min(column_data)
            max_value = max(column_data)
            value_range = max_value - min_value

            if value_range == 0:
                for index in range(len(column_data)):
                    self.dataset[column_name][index] = 0.0
            else:
                for index in range(len(column_data)):
                    normalized_value = (self.dataset[column_name][index] - min_value) / value_range
                    self.dataset[column_name][index] = float(normalized_value)

    def standard_scaler(self, columns: Set[str] = None):
        """ Aplica a padronização Z-score ($X_{std} = \frac{X - \mu}{\sigma}$) nas colunas especificadas. Modifica o dataset."""
        target_columns = self._get_target_columns(columns)
        for column_name in target_columns:
            if not self._validate_numeric_data(column_name):
                continue

            column_data = self.dataset[column_name]
            if not column_data:
                continue

            statistics_calculator = Statistics({column_name: column_data})
            mean_value = statistics_calculator.mean(column_name)
            std_deviation = statistics_calculator.stdev(column_name)

            if std_deviation == 0:
                for index in range(len(column_data)):
                    self.dataset[column_name][index] = 0.0
            else:
                for index in range(len(column_data)):
                    standardized_value = (self.dataset[column_name][index] - mean_value) / std_deviation
                    self.dataset[column_name][index] = float(standardized_value)

class Encoder:
    """Aplica codificação em colunas categóricas."""
    def __init__(self, dataset: Dict[str, List[Any]]):
        self.dataset = dataset

    def label_encode(self, columns: Set[str]):
        """ Converte cada categoria em uma coluna em um número inteiro. Modifica o dataset. """
        if not columns:
            return

        for column_name in columns:
            if column_name not in self.dataset:
                continue

            unique_categories = sorted(list(set(self.dataset[column_name])))
            category_to_number_mapping = {category: index for index, category in enumerate(unique_categories)}

            encoded_values = [category_to_number_mapping[value] for value in self.dataset[column_name]]
            self.dataset[column_name] = encoded_values

    def oneHot_encode(self, columns: Set[str]):
        """ Cria novas colunas binárias para cada categoria nas colunas especificadas (One-Hot Encoding).
        Modifica o dataset adicionando e removendo colunas."""
        if not columns:
            return

        new_columns_to_add = {}
        columns_to_remove = set()

        for column_name in columns:
            if column_name not in self.dataset:
                continue

            unique_categories = sorted(list(set(self.dataset[column_name])))

            for category in unique_categories:
                new_column_name = f'{column_name}_{category}'
                binary_values = [1 if value == category else 0 for value in self.dataset[column_name]]
                new_columns_to_add[new_column_name] = binary_values

            columns_to_remove.add(column_name)

        self.dataset.update(new_columns_to_add)
        for column_name in columns_to_remove:
            if column_name in self.dataset:
                del self.dataset[column_name]

class Preprocessing:
    """Classe principal que orquestra as operações de pré-processamento de dados."""
    def __init__(self, dataset: Dict[str, List[Any]]):
        if not isinstance(dataset, dict):
            raise TypeError("Dataset deve ser um dicionário")

        self.dataset = dataset
        self._validate_dataset_shape()
        
        # Atributos compostos para cada tipo de tarefa
        self.statistics = Statistics(self.dataset)
        self.missing_values = MissingValueProcessor(self.dataset)
        self.scaler = Scaler(self.dataset)
        self.encoder = Encoder(self.dataset)

    def _validate_dataset_shape(self):
        """Valida se todas as listas (colunas) no dicionário do dataset têm o mesmo comprimento."""
        if not self.dataset:
            return

        first_key = next(iter(self.dataset))
        expected_len = len(self.dataset[first_key])

        for key, value in self.dataset.items():
            if len(value) != expected_len:
                raise ValueError(
                    f"As colunas do dataset devem ter o mesmo comprimento. A coluna '{key}' tem {len(value)} elementos, mas o esperado era {expected_len}.")

    def isna(self, columns: Set[str] = None) -> Dict[str, List[Any]]:
        """Atalho para missing_values.isna(). Retorna as linhas com valores nulos."""
        return self.missing_values.isna(columns=columns)

    def notna(self, columns: Set[str] = None) -> Dict[str, List[Any]]:
        """Atalho para missing_values.notna(). Retorna as linhas sem valores nulos."""
        return self.missing_values.notna(columns=columns)

    def fillna(self, columns: Set[str] = None, method: str = 'mean', default_value: Any = 0):
        """ Atalho para missing_values.fillna(). Preenche valores nulos. Retorna 'self' para permitir encadeamento de métodos."""
        self.missing_values.fillna(columns=columns, method=method, default_value=default_value)
        return self

    def dropna(self, columns: Set[str] = None):
        """ Atalho para missing_values.dropna(). Remove linhas com valores nulos. Retorna 'self' para permitir encadeamento de métodos."""
        self.missing_values.dropna(columns=columns)
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
            self.scaler.minMax_scaler(columns=columns)
        elif method == 'standard':
            self.scaler.standard_scaler(columns=columns)
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
            self.encoder.label_encode(columns=columns)
        elif method == 'oneHot':
            self.encoder.oneHot_encode(columns=columns)
        else:
            raise ValueError(f"Método de codificação '{method}' não suportado. Use 'label' ou 'oneHot'.")
        return self
