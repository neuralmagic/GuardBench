import datasets

from .bibtex import jbb_behaviors
from .dataset import Dataset


class JBB_Behaviors(Dataset):
    def __init__(self):
        # Metadata
        self.name = "JBB Behaviors"
        self.target_role = "user"
        self.category = "prompts"
        self.subcategory = "instructions"
        self.languages = ["English"]
        self.sources = []
        self.license = "MIT"
        self.bibtex = jbb_behaviors
        self.alias = "jbb_behaviors"
        self.splits = ["test"]
        self.generation_methods = ["real-world"]
        self.hazard_categories = [
            "Disinformation",
            "Economic harm",
            "Expert advice",
            "Fraud/Deception",
            "Government decision-making",
            "Harassment/Discrimination",
            "Malware/Hacking",
            "Physical harm",
            "Privacy",
            "Sexual/Adult content",
        ]

        # Other
        self.repo_id = "JailbreakBench/JBB-Behaviors"

        # Init super object
        super().__init__(self.alias)

    def download_raw_data(self):
        self.tmp_path.mkdir(parents=True, exist_ok=True)
        datasets.load_dataset(self.repo_id, "behaviors")

    def add_id_prefix(self, x: dict, prefix: str) -> dict:
        return x | {"id": prefix + "_" + str(x["Index"])}

    def add_label(self, x: dict, label: str) -> dict:
        return x | {"label": label}

    def load_raw_data(self) -> dict[str, list[dict]]:
        safe_set = datasets.load_dataset(self.repo_id, "behaviors", split="benign")
        safe_set = [self.add_id_prefix(x, "safe") for x in safe_set]
        safe_set = [self.add_label(x, False) for x in safe_set]

        unsafe_set = datasets.load_dataset(self.repo_id, "behaviors", split="harmful")
        unsafe_set = [self.add_id_prefix(x, "unsafe") for x in unsafe_set]
        unsafe_set = [self.add_label(x, True) for x in unsafe_set]

        test_set = safe_set + unsafe_set

        return {"test": test_set}

    def format_sample(self, x: dict) -> dict:
        return dict(
            id=x["id"],
            label=x["label"],
            conversation=self.format_prompt_as_conversation(x["Goal"]),
            behavior=x["Behavior"],
            category=x["Category"],
            source=x["Source"],
        )

    def format_data(self, dataset: dict[str, list[dict]]) -> dict[str, list[dict]]:
        test_set = [self.format_sample(x) for x in dataset["test"]]

        return {"test": test_set}
