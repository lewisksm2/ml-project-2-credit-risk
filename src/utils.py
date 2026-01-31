import numpy as np
import matplotlib.pyplot as plt



def plot_numeric_distributions(df, n_cols=3, bins=30):
    
    num_cols = df.select_dtypes(include=np.number).columns
    
    n_rows = int(np.ceil(len(num_cols)/n_cols))
    
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize = (5*n_cols, 4*n_rows)
    )
    
    axes = axes.flatten()
    
    for i, col in enumerate(num_cols):
        axes[i].hist(df[col], bins=bins)
        axes[i].set_title(col)
        
    for j in range(len(num_cols), len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    plt.show()