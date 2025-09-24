# Chi-squared test
import numpy as np
from scipy.stats import chi2_contingency
import pandas as pd

df=pd.read_csv("confusion2.csv",index_col=0)
chi2, p, dof, expected = chi2_contingency(df)
n = df.sum().sum()  # Total provinces
min_dim = min(df.shape) - 1  # For Cramer’s V

# Cramer’s V (effect size)
cramers_v = np.sqrt(chi2 / (n * min_dim))

print(f"Chi-Squared = {chi2:.2f}, p = {p:.4f}, Cramer’s V = {cramers_v:.2f}")


## DEEPSEEK IMPLEMENTATION
import pandas as pd
#import rpy2.robjects as ro
#from rpy2.robjects.packages import importr

# Load R's 'stats' package
# stats = importr('stats')
#
# # Read and preprocess confusion matrices
# def load_table(file_path):
#     df = pd.read_csv(file_path, index_col=0)
#     df = df.drop(columns=['Total'])  # Remove 'Total' column
#     df = df.drop(index='Total') if 'Total' in df.index else df  # Remove 'Total' row
#     return df.values.tolist()  # Convert to R-friendly list
#
# # Perform FET with Monte Carlo simulation (10,000 replicates)
# def run_fisher(table, replicates=10000):
#     r_table = ro.r.matrix(table, nrow=len(table), ncol=len(table[0]))
#     result = stats.fisher_test(r_table, simulate_p_value=True, B=replicates)
#     return result[0][0]  # Extract p-value
#
# # Example usage
# confusion1_table = load_table('confusion1.csv')
# confusion2_table = load_table('confusion2.csv')

#p1 = run_fisher(confusion1_table)
#p2 = run_fisher(confusion2_table)

#print(f"FET p-value for confusion1.csv: {p1:.5f}")
#print(f"FET p-value for confusion2.csv: {p2:.5f}")

## GROK IMPLEMENTATION
import pandas as pd
import pingouin as pg
import numpy as np
from scipy.stats import chi2_contingency


# Load the contingency tables
def load_contingency_table(filename):
    df = pd.read_csv(filename, index_col=0)
    # Drop the 'Total' column if present
    if 'Total' in df.columns:
        df = df.drop(columns=['Total'])
    # Convert to numpy array for analysis
    data = df.to_numpy()
    return data, df.index.tolist(), df.columns.tolist()


# Function to perform Fisher's Exact Test and calculate Cramér's V
def analyze_contingency_table(data, row_labels, col_labels, election_year):
    # Fisher's Exact Test (using simulation for 3x3 table)
    fisher_result = pg.contingency.fisher_exact(data, alternative='two-sided')
    p_value = fisher_result['p-val']

    # Chi-square test for Cramér's V (since Fisher doesn't provide effect size directly)
    chi2, p_chi2, dof, expected = chi2_contingency(data)
    n = np.sum(data)
    min_dim = min(data.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim))

    # Print results
    print(f"\nAnalysis for {election_year} Election Results:")
    print(f"Contingency Table:")
    print(pd.DataFrame(data, index=row_labels, columns=col_labels))
    print(f"Fisher's Exact Test: p-value = {p_value:.4f}")
    print(f"Chi-square Test: chi^2 = {chi2:.2f}, p-value = {p_chi2:.4f}, dof = {dof}")
    print(f"Cramér's V = {cramers_v:.3f}")
    print(f"Expected Frequencies:\n{pd.DataFrame(expected, index=row_labels, columns=col_labels)}")


# Load and analyze confusion1.csv (2023 Parliamentary Elections)
data1, rows1, cols1 = load_contingency_table('confusion1.csv')
analyze_contingency_table(data1, rows1, cols1, "2023 Parliamentary")

# Load and analyze confusion2.csv (2024 Local Elections)
data2, rows2, cols2 = load_contingency_table('confusion2.csv')
analyze_contingency_table(data2, rows2, cols2, "2024 Local")

# Hypotheses
print("\nHypotheses:")
print("- H_0: No association exists between naming clusters and political alliances (independence).")
print("- H_1: A significant association exists between naming clusters and political alliances.")
print("Conclusion: If p-value < 0.05, reject H_0 in favor of H_1.")