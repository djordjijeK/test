import re
import numpy as np
import pandas as pd

from pandas.api.types import is_string_dtype, is_numeric_dtype, is_categorical_dtype


def convert_to_categorical(data_frame, ordinal_categorical_features = None, drop_features = None, skip_features = None):
    """
    Converts all string columns in a panda's data frame to a column of categorical values.
    
    Parameters:
    -----------
    data_frame          : A pandas dataframe.
    ordinal_categorical : A string or list of strings that represent the name of ordinal categorical features. 
    drop_features       : A string or list of strings that represent the name of features to drop.
    skip_features       : A string or list of strings that represent the name of features to skip during processing.
    
    Examples:
    ---------
    >>> data_frame = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'a'], 'col3': ['Low', 'Medium', 'High']})
    >>> data_frame
           col1 col2    col3
        0     1    a     Low
        1     2    b  Medium
        2     3    a    High
        note that types of 'col2' and 'col3' are strings.
    >>> data_frame = convert_to_categorical(data_frame, drop_features = 'col1', skip_features = 'col2')
    >>> data_frame
          col2    col3
        0    a     Low
        1    b  Medium
        2    a    High
        note that type of 'col2' is string but the type of 'col3' is category.
        
    Returns:
    --------
    data_frame : The processed data frame.
    """
    df = data_frame.copy()
    
    if isinstance(ordinal_categorical_features, str):
        ordinal_categorical_features = [ordinal_categorical_features]
    
    if isinstance(drop_features, str):
        drop_features = [drop_features]
        
    if isinstance(skip_features, str):
        skip_features = [skip_features]
        
    if ordinal_categorical_features is None:
        ordinal_categorical_features = []
        
    if drop_features is None:
        drop_features = []
        
    if skip_features is None:
        skip_features = []
        
    df = df.drop(drop_features, axis = 1)
    
    for feature in df.columns:
        if feature not in skip_features and is_string_dtype(df[feature]):
            df[feature] = df[feature].astype('category').cat.as_ordered() if feature in ordinal_categorical_features else df[feature].astype('category')
            
    return df


def numericalize_categories(data_frame, max_categories = 0, skip_features = None):
    """
    Converts categorical features to numbers. In the process it creates the mapping of categorical variables.
    
    Parameters:
    -----------
    data_frame     : A pandas dataframe.
    skip_features  : A string or list of strings that represent the name of features to skip during processing.
    max_categories : If the feature has more categories than max_num_categories then we convert it to codes, otherwise we one-hot encode it.
    
    Examples:
    ---------
    >>> data_frame = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'a'], 'col3': ['Low', 'Medium', 'High']})
    >>> data_frame
           col1 col2    col3
        0     1    a     Low
        1     2    b  Medium
        2     3    a    High
        note that types of 'col2' and 'col3' are strings.
    >>> data_frame = convert_to_categorical(data_frame)
    >>> data_frame, mapper = numericalize_categories(data_frame)
    >>> data_frame, mapper
           col1  col2  col3
        0     1     1     2
        1     2     2     3
        2     3     1     1
        {'col2': {0: 'a', 1: 'b'}, 'col3': {0: 'High', 1: 'Low', 2: 'Medium'}}
    >>> data_frame = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'a'], 'col3': ['Low', 'Medium', 'High']})
    >>> data_frame
           col1 col2    col3
        0     1    a     Low
        1     2    b  Medium
        2     3    a    High
    >>> data_frame = convert_to_categorical(data_frame)
    >>> data_frame, mapper = numericalize_categories(data_frame, max_categories = 3)
    >>> data_frame, mapper
           col1  col2_a  col2_b  col3_High  col3_Low  col3_Medium
        0     1       1       0          0         1            0
        1     2       0       1          0         0            1
        2     3       1       0          1         0            0
        {'col2': 'one-hot', 'col3': 'one-hot'}

    Returns:
    --------
    data_frame      : The processed data frame.
    category_mapper : A mapping applied to each categorical variable.
    """
    df = data_frame.copy()
    category_mapper = dict()
    
    if isinstance(skip_features, str):
        skip_features = [skip_features]
        
    if skip_features is None:
        skip_features = []
        
    for feature in df.columns:
        if feature not in skip_features and is_categorical_dtype(df[feature]):
            if (len(df[feature].cat.categories) > max_categories):
                category_mapper[feature] = dict(enumerate(df[feature].cat.categories))
                df[feature] = df[feature].cat.codes + 1
            else:
                one_hot = pd.get_dummies(df[feature], prefix = f'{feature}')
                df = df.drop(feature, axis = 1)
                df = df.join(one_hot)
                category_mapper[feature] = 'one-hot'
                
    return df, category_mapper


def interpolate_missing_values(data_frame, skip_features = None):
    """
    Fill missing numeric data of a feature in a data frame with the median, and add a {feature}_missing column which specifies if the data was missing.
    
    Parameters:
    -----------
    data_frame    : A pandas dataframe.
    skip_features : A string or list of strings that represent the name of features to skip during processing.
    
    Examples:
    ---------
    >>> data_frame = pd.DataFrame({'col1': [1, np.nan, 3], 'col2': ['a', 'b', 'a'], 'col3': [np.nan, 2.5, 7.3]})
    >>> data_frame
           col1 col2  col3
        0   1.0    a   NaN
        1   NaN    b   2.5
        2   3.0    a   7.3
    >>> data_frame, mapper = structured.interpolate_missing_values(data_frame, skip_features = ['col1', 'col3'])
           col1 col2  col3  col1_missing  col3_missing
        0   1.0    a   4.9         False          True
        1   2.0    b   2.5          True         False
        2   3.0    a   7.3         False         False
    
    Returns:
    --------
    data_frame     : The processed data frame.
    numeric_mapper : A mapping applied to each numeric variable.
    """
    df = data_frame.copy()
    numeric_mapper = dict()
    
    if isinstance(skip_features, str):
        skip_features = [skip_features]
        
    if skip_features is None:
        skip_features = []
    
    for feature in df.columns:
        if feature not in skip_features and is_numeric_dtype(df[feature]):
            df[f'{feature}_missing'] = df[feature].isnull()
            df[feature] = df[feature].fillna(value = df[feature].median())
            numeric_mapper[feature] = df[feature].median()
            
    return df, numeric_mapper


def extract_date_features(data_frame, date_features, time = False, drop = True):
    """
    Converts a column(s) of a data frame from a datetime64 to many columns containing the information from the date feature.
    
    Parameters:
    -----------
    data_frame    : A pandas data frame.
    data_features : A string or list of strings that represent the name of the date column you wish to extract features from. If it is not a datetime64 series, it will be converted to one.
    time          : If true time features: hour, minute, second will be included.
    drop          : If true then the original date column will be removed.
    
    Examples:
    ---------
    >>> data_frame = pd.DataFrame({'A': ['3/11/2000', '3/12/2000', '3/13/2000']})
    >>> data_frame
                   A
        0  3/11/2000
        1  3/12/2000
        2  3/13/2000
    >>> data_frame = extract_features_from_date(data_frame, 'A')
    >>> data_frame
           A_year  A_month  A_week  A_day  A_dayofweek  A_dayofyear  A_is_month_end  A_is_month_start  A_is_quarter_end  A_is_quarter_start  A_is_year_end  A_is_year_start
        0    2000        3      10     11            5           71           False             False             False               False          False            False
        1    2000        3      10     12            6           72           False             False             False               False          False            False
        2    2000        3      11     13            0           73           False             False             False               False          False            False
        
    Returns:
    --------
    data_frame : A data frame with extracted features from date column(s).
    """
    df = data_frame.copy()

    attrs = ['year', 'month', 'week', 'day', 
            'dayofweek', 'dayofyear', 'is_month_end', 'is_month_start', 
            'is_quarter_end', 'is_quarter_start', 'is_year_end', 'is_year_start']
    
    if time:
        attrs = attrs + ['hour', 'minute', 'second']
        
    if isinstance(date_features, str):
        date_features = [date_features]

    for feature in date_features:
        column       = df[feature]
        column_dtype = df[feature].dtype 
        
        if isinstance(column.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
            column_dtype = np.datetime64
            
        if not np.issubdtype(column_dtype, np.datetime64):
            df[feature] = df[feature].astype(np.datetime64)
            
        name = re.sub('[Dd]ate$', '', feature)       
            
        for attr in attrs:
            df[f'{name}_{attr}'] = getattr(df[feature].dt, attr)
            
        if drop:
            df = df.drop(feature, axis = 1)
    
    return df