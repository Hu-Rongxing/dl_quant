from load_data.pandas_object_data import generate_wide_dataframe


if __name__ == '__main__':
    target_df, covariate_df = generate_wide_dataframe()
    target_df.to_csv('data/target.csv')
    covariate_df.to_csv('data/covariate.csv')
    print(target_df.head())
    print(covariate_df.head())