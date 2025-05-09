import pandas as pd

def reshape_sales_data(df, id_vars, calendar_df=None):
    """
    Convert wide-format sales DataFrame into long-format with one row per item_id, store_id, date, and sales.
    
    Parameters:
    - df (pd.DataFrame): Input DataFrame with 'd_1' to 'd_1913' columns representing daily sales.
    - calendar_df (pd.DataFrame, optional): DataFrame containing columns ['d', 'date'] to map 'd_1' to actual dates.
    
    Returns:
    - pd.DataFrame: Long-format DataFrame with columns ['item_id', 'store_id', 'date', 'sales']
    """
    value_vars = [col for col in df.columns if col.startswith('d_')]

    # Melt the DataFrame
    melted_df = df.melt(
        id_vars=id_vars, 
        value_vars=value_vars,
        var_name='d', 
        value_name='sales')

    # Merge with calendar if provided
    if calendar_df is not None:
        if not {'d', 'date'}.issubset(calendar_df.columns):
            raise ValueError("calendar_df must contain 'd' and 'date' columns.")
        melted_df = melted_df.merge(calendar_df[['d', 'date']], on='d', how='left')
    else:
        melted_df['date'] = melted_df['d']  # Fallback to 'd' if no calendar provided

    return melted_df[id_vars+['date', 'sales']].sort_values(by=id_vars+['date'])
