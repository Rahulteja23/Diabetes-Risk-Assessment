# ============================================================
# DIABETES RISK ASSESSMENT
# LOGISTIC REGRESSION + BAYESIAN NETWORK
# ============================================================

import itertools
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

TARGET = "Outcome"

ZERO_AS_MISSING = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]


# ============================================================
# 1. LOAD DATASET
# ============================================================

def load_dataset(csv_file_path):

    print("\n" + "=" * 75)
    print("STEP 1: LOADING DATASET")
    print("=" * 75)

    print("\nLoading:", csv_file_path)

    df = pd.read_csv(csv_file_path)

    if len(df.columns) == 9:

        expected_columns = FEATURES + [TARGET]

        try:

            pd.to_numeric(df.iloc[0])

            df.columns = expected_columns

            print(
                "\nColumn names were automatically assigned."
            )

        except Exception:
            pass

    required_columns = FEATURES + [TARGET]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "\nMissing required columns:\n"
            + str(missing_columns)
        )

    df = df[
        required_columns
    ].copy()

    for column in required_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna(
        subset=[TARGET]
    )

    print(
        "\nDataset loaded successfully."
    )

    print(
        "Dataset shape:",
        df.shape
    )

    print("\nFirst 5 rows:")

    display(
        df.head()
    )

    print("\nTarget distribution:")

    print(
        df[TARGET].value_counts()
    )

    return df


# ============================================================
# 2. PREPROCESSING
# ============================================================

def preprocess_dataset(df):

    print("\n" + "=" * 75)
    print("STEP 2: DATA PREPROCESSING")
    print("=" * 75)

    df = df.copy()

    print(
        "\nReplacing invalid zero values with NaN..."
    )

    for column in ZERO_AS_MISSING:

        df[column] = df[column].replace(
            0,
            np.nan
        )

    print(
        "\nMissing values BEFORE imputation:"
    )

    print(
        df.isnull().sum()
    )

    print(
        "\nPerforming Random Forest iterative imputation..."
    )

    rf_estimator = RandomForestRegressor(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        min_samples_leaf=2
    )

    imputer = IterativeImputer(
        estimator=rf_estimator,
        max_iter=10,
        random_state=RANDOM_STATE,
        initial_strategy="median"
    )

    df[FEATURES] = imputer.fit_transform(
        df[FEATURES]
    )

    print(
        "\nMissing values AFTER imputation:"
    )

    print(
        df.isnull().sum()
    )

    print(
        "\nPreprocessing completed."
    )

    return df


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

def split_dataset(df):

    print("\n" + "=" * 75)
    print("STEP 3: TRAIN / TEST SPLIT")
    print("=" * 75)

    train_df, test_df = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=df[TARGET]
    )

    print(
        "\nTraining samples:",
        len(train_df)
    )

    print(
        "Testing samples :",
        len(test_df)
    )

    print(
        "\nTraining outcome distribution:"
    )

    print(
        train_df[TARGET].value_counts()
    )

    print(
        "\nTesting outcome distribution:"
    )

    print(
        test_df[TARGET].value_counts()
    )

    return train_df, test_df


# ============================================================
# 4. LOGISTIC REGRESSION
# ============================================================

def train_logistic_regression(train_df):

    print("\n" + "=" * 75)
    print("STEP 4: TRAINING LOGISTIC REGRESSION")
    print("=" * 75)

    X_train = train_df[
        FEATURES
    ]

    y_train = train_df[
        TARGET
    ]

    model = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),

        (
            "logistic_regression",
            LogisticRegression(
                max_iter=5000,
                random_state=RANDOM_STATE
            )
        )
    ])

    model.fit(
        X_train,
        y_train
    )

    print(
        "\nLogistic Regression training completed."
    )

    return model


# ============================================================
# 5. LOGISTIC REGRESSION ROC
# ============================================================

def evaluate_logistic_regression(
    model,
    train_df,
    test_df
):

    print("\n" + "=" * 75)
    print("STEP 5: LOGISTIC REGRESSION ROC-AUC")
    print("=" * 75)

    train_probabilities = model.predict_proba(
        train_df[FEATURES]
    )[:, 1]

    test_probabilities = model.predict_proba(
        test_df[FEATURES]
    )[:, 1]

    train_auc = roc_auc_score(
        train_df[TARGET],
        train_probabilities
    )

    test_auc = roc_auc_score(
        test_df[TARGET],
        test_probabilities
    )

    train_fpr, train_tpr, _ = roc_curve(
        train_df[TARGET],
        train_probabilities
    )

    test_fpr, test_tpr, _ = roc_curve(
        test_df[TARGET],
        test_probabilities
    )

    print(
        "\nLogistic Regression:"
    )

    print(
        f"Training AUC: {train_auc:.4f}"
    )

    print(
        f"Testing AUC : {test_auc:.4f}"
    )

    return {
        "train_auc": train_auc,
        "test_auc": test_auc,
        "train_fpr": train_fpr,
        "train_tpr": train_tpr,
        "test_fpr": test_fpr,
        "test_tpr": test_tpr,
        "train_probabilities": train_probabilities,
        "test_probabilities": test_probabilities
    }


# ============================================================
# 6. DISCRETIZATION FOR BAYESIAN NETWORK
# ============================================================

def discretize_dataset(df):

    discrete_df = pd.DataFrame(
        index=df.index
    )

    discrete_df["Pregnancies"] = pd.cut(
        df["Pregnancies"],
        bins=[
            -np.inf,
            3,
            6,
            np.inf
        ],
        labels=[
            "Low",
            "Medium",
            "High"
        ],
        include_lowest=True
    )

    discrete_df["Glucose"] = pd.cut(
        df["Glucose"],
        bins=[
            -np.inf,
            100,
            140,
            np.inf
        ],
        labels=[
            "Low",
            "Medium",
            "High"
        ],
        include_lowest=True
    )

    discrete_df["BloodPressure"] = pd.cut(
        df["BloodPressure"],
        bins=[
            -np.inf,
            60,
            90,
            np.inf
        ],
        labels=[
            "Low",
            "Normal",
            "High"
        ],
        include_lowest=True
    )

    discrete_df["SkinThickness"] = pd.cut(
        df["SkinThickness"],
        bins=[
            -np.inf,
            20,
            30,
            np.inf
        ],
        labels=[
            "Low",
            "Medium",
            "High"
        ],
        include_lowest=True
    )

    discrete_df["Insulin"] = pd.cut(
        df["Insulin"],
        bins=[
            -np.inf,
            100,
            200,
            np.inf
        ],
        labels=[
            "Low",
            "Medium",
            "High"
        ],
        include_lowest=True
    )

    discrete_df["BMI"] = pd.cut(
        df["BMI"],
        bins=[
            -np.inf,
            25,
            30,
            np.inf
        ],
        labels=[
            "Normal",
            "Overweight",
            "Obesity"
        ],
        include_lowest=True
    )

    discrete_df["DiabetesPedigreeFunction"] = pd.cut(
        df["DiabetesPedigreeFunction"],
        bins=[
            -np.inf,
            0.30,
            0.70,
            np.inf
        ],
        labels=[
            "Low",
            "Medium",
            "High"
        ],
        include_lowest=True
    )

    discrete_df["Age"] = pd.cut(
        df["Age"],
        bins=[
            -np.inf,
            35,
            65,
            np.inf
        ],
        labels=[
            "Young Adult",
            "Middle Adult",
            "Senior"
        ],
        include_lowest=True
    )

    if "Outcome" in df.columns:

        discrete_df["Diabetes"] = (
            df["Outcome"].map({
                0: "No",
                1: "Yes"
            })
        )

    elif "Diabetes" in df.columns:

        discrete_df["Diabetes"] = (
            df["Diabetes"]
        )

    for column in discrete_df.columns:

        discrete_df[column] = (
            discrete_df[column]
            .astype(str)
        )

    return discrete_df


# ============================================================
# 7. BAYESIAN NETWORK FUNCTIONS
# ============================================================

def get_parents(
    node,
    edges
):

    return [
        source
        for source, destination in edges
        if destination == node
    ]


def is_valid_dag(
    edges,
    nodes
):

    graph = nx.DiGraph()

    graph.add_nodes_from(
        nodes
    )

    graph.add_edges_from(
        edges
    )

    return nx.is_directed_acyclic_graph(
        graph
    )


# ============================================================
# 8. BIC SCORE
# ============================================================

def local_bic_score(
    data,
    node,
    parents
):

    n = len(data)

    node_states = data[
        node
    ].unique()

    r = len(
        node_states
    )

    if len(parents) == 0:

        counts = data[
            node
        ].value_counts()

        log_likelihood = 0.0

        for count in counts:

            if count > 0:

                probability = (
                    count / n
                )

                log_likelihood += (
                    count *
                    np.log(probability)
                )

        parameters = r - 1

        return (
            log_likelihood
            -
            0.5 *
            parameters *
            np.log(n)
        )

    log_likelihood = 0.0

    grouped = data.groupby(
        parents + [node]
    ).size()

    parent_counts = data.groupby(
        parents
    ).size()

    for index, count in grouped.items():

        if len(parents) == 1:

            parent_values = index[0]

        else:

            parent_values = index[:-1]

        try:

            parent_count = (
                parent_counts.loc[
                    parent_values
                ]
            )

        except:

            continue

        if count > 0 and parent_count > 0:

            probability = (
                count /
                parent_count
            )

            log_likelihood += (
                count *
                np.log(probability)
            )

    number_of_parent_configurations = 1

    for parent in parents:

        number_of_parent_configurations *= (
            data[parent].nunique()
        )

    parameters = (
        r - 1
    ) * number_of_parent_configurations

    return (
        log_likelihood
        -
        0.5 *
        parameters *
        np.log(n)
    )


# ============================================================
# 9. NETWORK BIC
# ============================================================

def calculate_network_bic(
    data,
    edges
):

    total_score = 0.0

    for node in data.columns:

        parents = get_parents(
            node,
            edges
        )

        total_score += (
            local_bic_score(
                data,
                node,
                parents
            )
        )

    return total_score


# ============================================================
# 10. BAYESIAN NETWORK LEARNING
#
# ONLY CHANGE:
# CONTINUE ADDING EDGES UNTIL 15 EDGES ARE OBTAINED
# ============================================================

def learn_bayesian_network(
    data,
    max_iterations=50,
    tabu_size=15,
    max_parents=3,
    target_edges=15
):

    print("\n" + "=" * 75)
    print("STEP 7: LEARNING BAYESIAN NETWORK")
    print("=" * 75)

    print(
        f"\nTarget number of edges: {target_edges}"
    )

    nodes = list(
        data.columns
    )

    current_edges = set()

    current_score = calculate_network_bic(
        data,
        current_edges
    )

    tabu_list = []

    # --------------------------------------------------------
    # Keep adding edges until target = 15
    # --------------------------------------------------------

    for iteration in range(
        max_iterations
    ):

        if len(current_edges) >= target_edges:

            break

        candidates = []

        # ====================================================
        # ADD EDGE
        # ====================================================

        for source in nodes:

            for destination in nodes:

                if source == destination:

                    continue

                edge = (
                    source,
                    destination
                )

                if edge in current_edges:

                    continue

                parents = get_parents(
                    destination,
                    current_edges
                )

                if len(parents) >= max_parents:

                    continue

                new_edges = set(
                    current_edges
                )

                new_edges.add(
                    edge
                )

                if not is_valid_dag(
                    new_edges,
                    nodes
                ):

                    continue

                candidate_score = (
                    calculate_network_bic(
                        data,
                        new_edges
                    )
                )

                candidates.append(
                    (
                        candidate_score,
                        edge,
                        new_edges
                    )
                )

        # ====================================================
        # NO VALID EDGE
        # ====================================================

        if len(candidates) == 0:

            print(
                "\nWARNING:"
            )

            print(
                "No more valid edges can be added."
            )

            break

        # ====================================================
        # SORT CANDIDATES BY BIC
        # ====================================================

        candidates.sort(
            key=lambda x: x[0],
            reverse=True
        )

        selected_candidate = None

        # ====================================================
        # SELECT BEST NON-TABU EDGE
        # ====================================================

        for candidate in candidates:

            candidate_score = candidate[0]

            candidate_edge = candidate[1]

            move = (
                "ADD",
                candidate_edge
            )

            if move in tabu_list:

                continue

            selected_candidate = candidate

            break

        # ====================================================
        # IF ALL ARE TABU
        # ====================================================

        if selected_candidate is None:

            selected_candidate = (
                candidates[0]
            )

        (
            candidate_score,
            selected_edge,
            selected_edges
        ) = selected_candidate

        # ====================================================
        # ADD THE EDGE
        # ====================================================

        current_edges = set(
            selected_edges
        )

        current_score = (
            candidate_score
        )

        tabu_list.append(
            (
                "ADD",
                selected_edge
            )
        )

        if len(tabu_list) > tabu_size:

            tabu_list.pop(0)

        # ====================================================
        # DISPLAY
        # ====================================================

        print(
            f"Iteration {iteration + 1:02d} | "
            f"BIC = {current_score:.2f} | "
            f"Edges = {len(current_edges):02d} | "
            f"Added = "
            f"{selected_edge[0]} --> "
            f"{selected_edge[1]}"
        )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    print("\n" + "-" * 75)

    print(
        f"Final number of edges: "
        f"{len(current_edges)}"
    )

    print(
        f"Final BIC score: "
        f"{current_score:.4f}"
    )

    print(
        "\nLearned Bayesian Network edges:"
    )

    for number, (
        source,
        destination
    ) in enumerate(
        sorted(current_edges),
        start=1
    ):

        print(
            f"{number:02d}. "
            f"{source} --> {destination}"
        )

    print("-" * 75)

    if len(current_edges) == target_edges:

        print(
            f"\nSUCCESS: Bayesian Network "
            f"contains exactly "
            f"{target_edges} edges."
        )

    else:

        print(
            f"\nWARNING: "
            f"Only {len(current_edges)} edges "
            f"could be constructed."
        )

    return (
        list(current_edges),
        current_score
    )


# ============================================================
# 11. CREATE CPTs
# ============================================================

def create_probability_tables(
    data,
    edges,
    laplace=1.0
):

    print("\n" + "=" * 75)
    print("STEP 8: CREATING BAYESIAN NETWORK CPTs")
    print("=" * 75)

    states = {}

    for column in data.columns:

        states[column] = sorted(
            data[column].unique()
        )

    parents_dictionary = {}

    for node in data.columns:

        parents_dictionary[node] = (
            get_parents(
                node,
                edges
            )
        )

    probability_tables = {}

    for node in data.columns:

        parents = parents_dictionary[
            node
        ]

        table = {}

        if len(parents) == 0:

            counts = data[
                node
            ].value_counts()

            total = (
                len(data)
                +
                laplace *
                len(states[node])
            )

            for state in states[node]:

                probability = (
                    counts.get(
                        state,
                        0
                    )
                    +
                    laplace
                ) / total

                table[
                    (state,)
                ] = probability

        else:

            parent_state_lists = [
                states[parent]
                for parent in parents
            ]

            combinations = itertools.product(
                *parent_state_lists
            )

            for parent_values in combinations:

                mask = np.ones(
                    len(data),
                    dtype=bool
                )

                for (
                    parent,
                    value
                ) in zip(
                    parents,
                    parent_values
                ):

                    mask &= (
                        data[parent].values
                        ==
                        value
                    )

                subset = data.loc[
                    mask,
                    node
                ]

                counts = (
                    subset.value_counts()
                )

                total = (
                    len(subset)
                    +
                    laplace *
                    len(states[node])
                )

                for state in states[node]:

                    probability = (
                        counts.get(
                            state,
                            0
                        )
                        +
                        laplace
                    ) / total

                    table[
                        tuple(parent_values)
                        +
                        (state,)
                    ] = probability

        probability_tables[
            node
        ] = table

    print(
        "\nCPT creation completed."
    )

    return (
        states,
        parents_dictionary,
        probability_tables
    )


# ============================================================
# 12. BAYESIAN INFERENCE
# ============================================================

def bayesian_inference(
    query_node,
    evidence,
    states,
    parents,
    probability_tables,
    nodes
):

    hidden_nodes = [
        node
        for node in nodes
        if (
            node not in evidence
            and
            node != query_node
        )
    ]

    probabilities = {}

    for query_state in states[
        query_node
    ]:

        assignments = [
            {
                **evidence,
                query_node: query_state
            }
        ]

        for hidden_node in hidden_nodes:

            new_assignments = []

            for assignment in assignments:

                for state in states[
                    hidden_node
                ]:

                    new_assignment = dict(
                        assignment
                    )

                    new_assignment[
                        hidden_node
                    ] = state

                    new_assignments.append(
                        new_assignment
                    )

            assignments = new_assignments

        total_probability = 0.0

        for assignment in assignments:

            joint_probability = 1.0

            for node in nodes:

                parent_values = [
                    assignment[parent]
                    for parent in parents[node]
                ]

                key = (
                    tuple(parent_values)
                    +
                    (
                        assignment[node],
                    )
                )

                probability = (
                    probability_tables[
                        node
                    ].get(
                        key,
                        1e-12
                    )
                )

                joint_probability *= (
                    probability
                )

            total_probability += (
                joint_probability
            )

        probabilities[
            query_state
        ] = total_probability

    normalization = sum(
        probabilities.values()
    )

    if normalization == 0:

        return probabilities

    for state in probabilities:

        probabilities[state] /= (
            normalization
        )

    return probabilities


# ============================================================
# 13. PATIENT TO EVIDENCE
# ============================================================

def patient_to_evidence(
    patient
):

    patient_df = pd.DataFrame(
        [patient]
    )

    discrete_patient = (
        discretize_dataset(
            patient_df
        )
    )

    evidence = {}

    for feature in FEATURES:

        evidence[feature] = (
            discrete_patient.iloc[0][
                feature
            ]
        )

    return evidence


# ============================================================
# 14. BAYESIAN PREDICTIONS
# ============================================================

def get_bayesian_predictions(
    df,
    states,
    parents,
    probability_tables
):

    probabilities = []

    nodes = list(
        states.keys()
    )

    for _, row in df.iterrows():

        evidence = (
            patient_to_evidence(
                row
            )
        )

        posterior = bayesian_inference(
            query_node="Diabetes",
            evidence=evidence,
            states=states,
            parents=parents,
            probability_tables=probability_tables,
            nodes=nodes
        )

        probability_yes = posterior.get(
            "Yes",
            0.0
        )

        probabilities.append(
            probability_yes
        )

    return np.array(
        probabilities
    )


# ============================================================
# 15. BAYESIAN NETWORK ROC-AUC
# ============================================================

def evaluate_bayesian_network(
    train_df,
    test_df,
    states,
    parents,
    probability_tables
):

    print("\n" + "=" * 75)
    print("STEP 9: BAYESIAN NETWORK ROC-AUC")
    print("=" * 75)

    print(
        "\nCalculating Bayesian Network training probabilities..."
    )

    train_probabilities = (
        get_bayesian_predictions(
            train_df,
            states,
            parents,
            probability_tables
        )
    )

    print(
        "Calculating Bayesian Network testing probabilities..."
    )

    test_probabilities = (
        get_bayesian_predictions(
            test_df,
            states,
            parents,
            probability_tables
        )
    )

    train_auc = roc_auc_score(
        train_df[TARGET],
        train_probabilities
    )

    test_auc = roc_auc_score(
        test_df[TARGET],
        test_probabilities
    )

    train_fpr, train_tpr, _ = roc_curve(
        train_df[TARGET],
        train_probabilities
    )

    test_fpr, test_tpr, _ = roc_curve(
        test_df[TARGET],
        test_probabilities
    )

    print(
        "\nBayesian Network:"
    )

    print(
        f"Training AUC: {train_auc:.4f}"
    )

    print(
        f"Testing AUC : {test_auc:.4f}"
    )

    return {
        "train_auc": train_auc,
        "test_auc": test_auc,
        "train_fpr": train_fpr,
        "train_tpr": train_tpr,
        "test_fpr": test_fpr,
        "test_tpr": test_tpr,
        "train_probabilities": train_probabilities,
        "test_probabilities": test_probabilities
    }


# ============================================================
# 16. PAPER-STYLE ROC COMPARISON
# ============================================================

def create_roc_comparison(
    logistic_results,
    bayesian_results
):

    print("\n" + "=" * 75)
    print("STEP 10: CREATING ROC-AUC COMPARISON")
    print("=" * 75)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 6)
    )

    # ========================================================
    # A - LOGISTIC REGRESSION
    # ========================================================

    ax = axes[0]

    ax.plot(
        logistic_results["train_fpr"],
        logistic_results["train_tpr"],
        linewidth=2,
        label=(
            f"Training AUC = "
            f"{logistic_results['train_auc']:.3f}"
        )
    )

    ax.plot(
        logistic_results["test_fpr"],
        logistic_results["test_tpr"],
        linewidth=2,
        label=(
            f"Testing AUC = "
            f"{logistic_results['test_auc']:.3f}"
        )
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1
    )

    ax.set_title(
        "A  Logistic Regression",
        loc="left",
        fontsize=13,
        fontweight="bold"
    )

    ax.set_xlabel(
        "1 − specificity",
        fontsize=11
    )

    ax.set_ylabel(
        "Sensitivity",
        fontsize=11
    )

    ax.set_xlim(
        0,
        1
    )

    ax.set_ylim(
        0,
        1.05
    )

    ax.grid(
        alpha=0.2
    )

    ax.legend(
        loc="lower right",
        fontsize=9
    )

    # ========================================================
    # B - BAYESIAN NETWORK
    # ========================================================

    ax = axes[1]

    ax.plot(
        bayesian_results["train_fpr"],
        bayesian_results["train_tpr"],
        linewidth=2,
        label=(
            f"Training AUC = "
            f"{bayesian_results['train_auc']:.3f}"
        )
    )

    ax.plot(
        bayesian_results["test_fpr"],
        bayesian_results["test_tpr"],
        linewidth=2,
        label=(
            f"Testing AUC = "
            f"{bayesian_results['test_auc']:.3f}"
        )
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1
    )

    ax.set_title(
        "B  Bayesian Network",
        loc="left",
        fontsize=13,
        fontweight="bold"
    )

    ax.set_xlabel(
        "1 − specificity",
        fontsize=11
    )

    ax.set_ylabel(
        "Sensitivity",
        fontsize=11
    )

    ax.set_xlim(
        0,
        1
    )

    ax.set_ylim(
        0,
        1.05
    )

    ax.grid(
        alpha=0.2
    )

    ax.legend(
        loc="lower right",
        fontsize=9
    )

    plt.tight_layout()

    plt.savefig(
        "roc_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print(
        "\nROC comparison saved as:"
    )

    print(
        "roc_comparison.png"
    )


# ============================================================
# 17. LOGISTIC ROC
# ============================================================

def save_logistic_roc(results):

    plt.figure(
        figsize=(8, 6)
    )

    plt.plot(
        results["train_fpr"],
        results["train_tpr"],
        linewidth=2,
        label=(
            f"Training AUC = "
            f"{results['train_auc']:.3f}"
        )
    )

    plt.plot(
        results["test_fpr"],
        results["test_tpr"],
        linewidth=2,
        label=(
            f"Testing AUC = "
            f"{results['test_auc']:.3f}"
        )
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel(
        "1 − specificity"
    )

    plt.ylabel(
        "Sensitivity"
    )

    plt.title(
        "Logistic Regression ROC Curve"
    )

    plt.legend(
        loc="lower right"
    )

    plt.grid(
        alpha=0.2
    )

    plt.tight_layout()

    plt.savefig(
        "logistic_regression_roc.png",
        dpi=300
    )

    plt.show()


# ============================================================
# 18. BAYESIAN ROC
# ============================================================

def save_bayesian_roc(results):

    plt.figure(
        figsize=(8, 6)
    )

    plt.plot(
        results["train_fpr"],
        results["train_tpr"],
        linewidth=2,
        label=(
            f"Training AUC = "
            f"{results['train_auc']:.3f}"
        )
    )

    plt.plot(
        results["test_fpr"],
        results["test_tpr"],
        linewidth=2,
        label=(
            f"Testing AUC = "
            f"{results['test_auc']:.3f}"
        )
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel(
        "1 − specificity"
    )

    plt.ylabel(
        "Sensitivity"
    )

    plt.title(
        "Bayesian Network ROC Curve"
    )

    plt.legend(
        loc="lower right"
    )

    plt.grid(
        alpha=0.2
    )

    plt.tight_layout()

    plt.savefig(
        "bayesian_network_roc.png",
        dpi=300
    )

    plt.show()


# ============================================================
# 19. VISUALIZE BAYESIAN NETWORK
# ============================================================

def visualize_bayesian_network(edges):

    print("\n" + "=" * 75)
    print("STEP 11: BAYESIAN NETWORK STRUCTURE")
    print("=" * 75)

    graph = nx.DiGraph()

    graph.add_nodes_from(
        FEATURES + ["Diabetes"]
    )

    graph.add_edges_from(
        edges
    )

    print(
        "\nLearned Bayesian Network edges:"
    )

    for source, destination in sorted(
        edges
    ):

        print(
            f"{source} --> {destination}"
        )

    print(
        "\nTOTAL EDGES:",
        len(edges)
    )

    plt.figure(
        figsize=(14, 9)
    )

    positions = nx.spring_layout(
        graph,
        seed=RANDOM_STATE,
        k=1.2
    )

    nx.draw_networkx_nodes(
        graph,
        positions,
        node_size=3000
    )

    nx.draw_networkx_edges(
        graph,
        positions,
        arrows=True,
        arrowsize=25,
        width=2
    )

    nx.draw_networkx_labels(
        graph,
        positions,
        font_size=9
    )

    plt.title(
        f"Learned Bayesian Network "
        f"({len(edges)} Edges)"
    )

    plt.axis(
        "off"
    )

    plt.tight_layout()

    plt.savefig(
        "bayesian_network.png",
        dpi=300
    )

    plt.show()


# ============================================================
# 20. DIABETES RELATIONSHIPS
# ============================================================

def print_diabetes_relationships(edges):

    print("\n" + "=" * 75)
    print("STEP 12: DIABETES NETWORK RELATIONSHIPS")
    print("=" * 75)

    graph = nx.DiGraph(
        edges
    )

    parents = list(
        graph.predecessors(
            "Diabetes"
        )
    )

    ancestors = list(
        nx.ancestors(
            graph,
            "Diabetes"
        )
    )

    print(
        "\nDirect parents of Diabetes:"
    )

    if parents:

        for parent in parents:

            print(
                "  →",
                parent
            )

    else:

        print(
            "No direct parents found."
        )

    print(
        "\nAll upstream factors:"
    )

    if ancestors:

        for factor in ancestors:

            print(
                "  →",
                factor
            )

    else:

        print(
            "No upstream factors found."
        )


# ============================================================
# 21. MODEL COMPARISON
# ============================================================

def display_model_comparison(
    logistic_results,
    bayesian_results
):

    print("\n" + "=" * 75)
    print("MODEL PERFORMANCE COMPARISON")
    print("=" * 75)

    comparison = pd.DataFrame({

        "Model": [
            "Logistic Regression",
            "Bayesian Network"
        ],

        "Training AUC": [
            logistic_results["train_auc"],
            bayesian_results["train_auc"]
        ],

        "Testing AUC": [
            logistic_results["test_auc"],
            bayesian_results["test_auc"]
        ]

    })

    comparison["Training AUC"] = (
        comparison["Training AUC"].round(4)
    )

    comparison["Testing AUC"] = (
        comparison["Testing AUC"].round(4)
    )

    display(
        comparison
    )

    lr_auc = (
        logistic_results["test_auc"]
    )

    bn_auc = (
        bayesian_results["test_auc"]
    )

    print(
        "\nBest testing AUC:"
    )

    if lr_auc > bn_auc:

        print(
            f"Logistic Regression "
            f"({lr_auc:.4f})"
        )

    elif bn_auc > lr_auc:

        print(
            f"Bayesian Network "
            f"({bn_auc:.4f})"
        )

    else:

        print(
            "Both models have the same testing AUC."
        )


# ============================================================
# 22. PERSONALIZED BAYESIAN RISK
# ============================================================

def personalized_risk_prediction(
    states,
    parents,
    probability_tables
):

    print("\n" + "=" * 75)
    print("PERSONALIZED BAYESIAN RISK ASSESSMENT")
    print("=" * 75)

    print(
        "\nEnter the category for each feature."
    )

    print(
        "Use exactly one of the displayed categories."
    )

    evidence = {}

    for feature in FEATURES:

        available_states = states[
            feature
        ]

        print(
            f"\n{feature}"
        )

        print(
            "Available:",
            ", ".join(
                available_states
            )
        )

        value = input(
            f"Enter {feature}: "
        ).strip()

        if value not in available_states:

            print(
                "Invalid input."
            )

            print(
                "Using:",
                available_states[0]
            )

            value = available_states[0]

        evidence[
            feature
        ] = value

    posterior = bayesian_inference(
        query_node="Diabetes",
        evidence=evidence,
        states=states,
        parents=parents,
        probability_tables=probability_tables,
        nodes=list(states.keys())
    )

    print("\n" + "=" * 75)
    print("PERSONALIZED RISK RESULT")
    print("=" * 75)

    for state, probability in posterior.items():

        print(
            f"P(Diabetes = {state}) = "
            f"{probability:.4f}"
        )

    diabetes_probability = posterior.get(
        "Yes",
        0.0
    )

    print(
        "\nEstimated diabetes probability:",
        f"{diabetes_probability * 100:.2f}%"
    )

    if diabetes_probability >= 0.70:

        print(
            "Risk level: HIGH"
        )

    elif diabetes_probability >= 0.40:

        print(
            "Risk level: MODERATE"
        )

    else:

        print(
            "Risk level: LOW"
        )

    print(
        "\nNote: This is a machine-learning "
        "demonstration and not a medical diagnosis."
    )


# ============================================================
# 23. MAIN
# ============================================================

def main(csv_file_path):

    print("\n")
    print("=" * 75)
    print("DIABETES RISK ASSESSMENT")
    print("LOGISTIC REGRESSION + BAYESIAN NETWORK")
    print("=" * 75)

    # --------------------------------------------------------
    # STEP 1
    # --------------------------------------------------------

    df = load_dataset(
        csv_file_path
    )

    # --------------------------------------------------------
    # STEP 2
    # --------------------------------------------------------

    df = preprocess_dataset(
        df
    )

    # --------------------------------------------------------
    # STEP 3
    # --------------------------------------------------------

    train_df, test_df = (
        split_dataset(
            df
        )
    )

    # --------------------------------------------------------
    # STEP 4
    # --------------------------------------------------------

    logistic_model = (
        train_logistic_regression(
            train_df
        )
    )

    # --------------------------------------------------------
    # STEP 5
    # --------------------------------------------------------

    logistic_results = (
        evaluate_logistic_regression(
            logistic_model,
            train_df,
            test_df
        )
    )

    save_logistic_roc(
        logistic_results
    )

    # --------------------------------------------------------
    # STEP 6
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("STEP 6: PREPARING BAYESIAN NETWORK DATA")
    print("=" * 75)

    discrete_train_df = (
        discretize_dataset(
            train_df
        )
    )

    print(
        "\nDiscrete Bayesian Network data:"
    )

    display(
        discrete_train_df.head()
    )

    # --------------------------------------------------------
    # STEP 7
    # --------------------------------------------------------
    # IMPORTANT:
    # TARGET EDGES = 15
    # --------------------------------------------------------

    edges, bic_score = (
        learn_bayesian_network(
            discrete_train_df,
            max_iterations=50,
            tabu_size=15,
            max_parents=3,
            target_edges=15
        )
    )

    # --------------------------------------------------------
    # STEP 8
    # --------------------------------------------------------

    (
        states,
        parents,
        probability_tables
    ) = create_probability_tables(
        discrete_train_df,
        edges
    )
    print(create_probability_tables)
    # --------------------------------------------------------
    # STEP 9
    # --------------------------------------------------------

    bayesian_results = (
        evaluate_bayesian_network(
            train_df,
            test_df,
            states,
            parents,
            probability_tables
        )
    )

    save_bayesian_roc(
        bayesian_results
    )

    # --------------------------------------------------------
    # STEP 10
    # --------------------------------------------------------

    create_roc_comparison(
        logistic_results,
        bayesian_results
    )

    # --------------------------------------------------------
    # STEP 11
    # --------------------------------------------------------

    visualize_bayesian_network(
        edges
    )

    # --------------------------------------------------------
    # STEP 12
    # --------------------------------------------------------

    print_diabetes_relationships(
        edges
    )

    # --------------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------------

    display_model_comparison(
        logistic_results,
        bayesian_results
    )

    # --------------------------------------------------------
    # PERSONALIZED RISK
    # --------------------------------------------------------

    personalized_risk_prediction(
        states,
        parents,
        probability_tables
    )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print("\n")
    print("=" * 75)
    print("PROGRAM COMPLETED SUCCESSFULLY")
    print("=" * 75)
# ============================================================
# RUN PROGRAM
# ============================================================

csv_file_path = "diabetes.csv"

main(
    csv_file_path
)
