from typing import Optional, Tuple, Dict
import pandas as pd


def fetch_csv_data(url: str, separator: Optional[str]) -> pd.DataFrame:
    try:
        args = dict(filepath_or_buffer=url)
        if separator:
            args.update(sep=separator)
        return pd.read_csv(**args)
    except Exception as e:
        raise Exception(f'Error while fetching data at url {url}: {e}')


def build_train_test_sets(data: pd.DataFrame, label_col: str, train_size: float) -> \
        Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    try:
        from sklearn.model_selection import train_test_split

        train, test = train_test_split(data, train_size=train_size)
        x_y = lambda _data: (_data.drop([label_col], axis=1), _data[[label_col]])

        train_x, train_y = x_y(train)
        test_x, test_y = x_y(test)

        return dict(train=(train_x, train_y), test=(test_x, test_y))
    except Exception as e:
        raise Exception(f'Error while splitting data: {e}')
