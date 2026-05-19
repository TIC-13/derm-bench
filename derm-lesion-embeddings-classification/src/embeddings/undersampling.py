import pandas as pd

class UndersamplingUtils:
    @staticmethod
    def print_class_distribution(
        df: pd.DataFrame,
        header: str,
        label_col: str
    ) -> None:
        """Print the class distribution of a DataFrame.

        Args:
            df: Input DataFrame.
            header: Text prefix printed before the distribution.
            label_col: Name of the column containing class labels.
        """
        counts = df[label_col].value_counts(dropna=False)
        print(f"{header}: {counts.to_dict()} (total={len(df)})")

    @staticmethod
    def undersample_min_class(
        df: pd.DataFrame,
        label_col: str,
        random_state: int
    ) -> pd.DataFrame:
        """Undersample all classes to the size of the minority class.

        Args:
            df: Input DataFrame.
            label_col: Name of the column containing class labels.
            random_state: Random seed used for sampling.

        Returns:
            A balanced DataFrame where each class has the same number of
            samples as the original minority class.
        """
        min_n = df.groupby(label_col).size().min()
        return (
            df.groupby(label_col, group_keys=False)
            .sample(n=min_n, random_state=random_state)
            .reset_index(drop=True)
        )

    @staticmethod
    def undersample_max_per_class(
        df: pd.DataFrame,
        label_col: str,
        max_per_class: int,
        random_state: int
    ) -> pd.DataFrame:
        """Limit each class to a maximum number of samples.

        Args:
            df: Input DataFrame.
            label_col: Name of the column containing class labels.
            max_per_class: Maximum number of samples allowed per class.
            random_state: Random seed used for sampling.

        Returns:
            A DataFrame where each class has at most max_per_class samples.
        """
        groups = df.groupby(label_col, group_keys=False)
        return (
            groups
            .apply(lambda g: g.sample(
                n=min(len(g), max_per_class),
                random_state=random_state
            ))
            .reset_index(drop=True)
        )

    @staticmethod
    def undersample_ratio(
        df: pd.DataFrame,
        label_col: str,
        ratio: float,
        random_state: int
    ) -> pd.DataFrame:
        """Undersample each class based on a ratio of the minority class size.

        Args:
            df: Input DataFrame.
            label_col: Name of the column containing class labels.
            ratio: Multiplier applied to the minority class size.
            random_state: Random seed used for sampling.

        Returns:
            A DataFrame where each class has at most minority_size * ratio
            samples.
        """
        groups = df.groupby(label_col, group_keys=False)
        minority_n = groups.size().min()
        target_n = max(1, int(minority_n * ratio))

        return (
            groups
            .apply(lambda g: g.sample(
                n=min(len(g), target_n),
                random_state=random_state
            ))
            .reset_index(drop=True)
        )

    @staticmethod
    def apply_undersampling_if_enabled(
        df: pd.DataFrame,
        cfg: dict | None,
        label_col: str = "benign_malignant"
    ) -> pd.DataFrame:
        """Apply an undersampling strategy if enabled in the configuration.

        Args:
            df: Input DataFrame.
            cfg: Undersampling configuration dictionary. If None or disabled,
                the original DataFrame is returned.
            label_col: Name of the column containing class labels.

        Returns:
            The undersampled DataFrame if undersampling is enabled; otherwise,
            the original DataFrame.
        """
        if not cfg or not cfg.get("enabled", False):
            return df

        method = cfg.get("method", "min_class")
        random_state = int(cfg.get("random_state", 42))

        UndersamplingUtils.print_class_distribution(
            df, "[Undersampling] Before", label_col
        )

        if method == "min_class":
            df_bal = UndersamplingUtils.undersample_min_class(
                df, label_col, random_state
            )
        elif method == "max_per_class":
            df_bal = UndersamplingUtils.undersample_max_per_class(
                df,
                label_col,
                int(cfg.get("max_per_class", 1000)),
                random_state
            )
        elif method == "ratio":
            df_bal = UndersamplingUtils.undersample_ratio(
                df,
                label_col,
                float(cfg.get("ratio", 1.0)),
                random_state
            )
        else:
            print(
                f"[Undersampling] Unknown method '{method}'. "
                "Skipping undersampling."
            )
            return df

        UndersamplingUtils.print_class_distribution(
            df_bal, "[Undersampling] After", label_col
        )

        return df_bal
