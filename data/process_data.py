import sys
import pandas as pd
from sqlalchemy import create_engine 

def load_data(messages_filepath, categories_filepath):
    """
    Loads and merges data from messenges and categories file (.csv).
    
    Args:
    messages_filepath: path of messenges.csv file
    categories_filepath:  path of categories.csv file
    
    Returns:
    df: df of messages and categories merged
    
    """
    # load messenges data using pandas
    df_messages = pd.read_csv(messages_filepath)
    # load categories data using pandas
    df_categories = pd.read_csv(categories_filepath)
    # merge datasets 
    df = pd.merge(left=df_messages, right=df_categories, on="id")
    return df


def clean_data(df):
    """
    preprocess data.
    
    Args:
    df: dataFrame

    Returns:
    df: Dataframe after preprocessed
    """

    # create a dataframe of each category columns
    categories = df.categories.str.split(';', expand=True)
    # Change data colum's name
    categories.columns = categories.iloc[0].apply(lambda x: x[:-2]).values
    # Convert category value to 0 or 1 (values 2 also convert to 1)
    categories = categories.applymap(lambda x: int(x[-1] != '0'))
    # drop the original categories column from df
    df.drop(columns='categories', inplace=True)
    # Concatenate df and categories data frames
    df = pd.concat([df, categories], axis=1)
    # drop the original categories column from df
    df.drop_duplicates(inplace=True)
    return df


def save_data(df, database_filename):
    """
    Save data in SQLite database.

    Args:
    df: data need to save
    database_filename: database file name where to save
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('disaster_data', engine, index=False, if_exists='replace')    


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()