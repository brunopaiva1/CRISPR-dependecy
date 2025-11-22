import pandas as pd
import numpy as np

print("1. Carregando datasets...")

df_expression = pd.read_csv('data/OmicsExpressionTPMLogp1HumanProteinCodingGenesStranded.csv', low_memory=False)
df_dependency = pd.read_csv('data/CRISPRGeneDependency.csv', index_col=0)

if 'ModelID' in df_expression.columns:
    df_expression = df_expression.set_index('ModelID')

df_expression = df_expression[~df_expression.index.duplicated(keep='first')]

print("Filtrando apenas colunas numéricas...")
df_expression = df_expression.select_dtypes(include=['float32', 'float64', 'int32', 'int64', 'number'])

print(f"Shape Expressão (Apenas Números): {df_expression.shape}")

common_cells = df_expression.index.intersection(df_dependency.index)
print(f"Células comuns (Interseção): {len(common_cells)}")

X_data = df_expression.loc[common_cells]
Y_data = df_dependency.loc[common_cells]

gene_name = "BRAF"
target_cols = [col for col in Y_data.columns if gene_name in col.split(" ")[0]]
if not target_cols:
    raise ValueError(f"Gene {gene_name} não encontrado no dataset de dependência.")
target_column = target_cols[0]

print(f"Alvo selecionado: {target_column}")

X = np.nan_to_num(X_data.values.astype(np.float32))
y = np.nan_to_num(Y_data[target_column].values.astype(np.float32))

print("\n--- DADOS PRONTOS ---")
print(f"X (Features): {X.shape}")
print(f"y (Target): {y.shape}")

assert X.shape[0] == y.shape[0], "ERRO CRÍTICO: X e y têm números diferentes de linhas!"
