"""Loader for the CONFAIDE benchmark (Mireshghallah et al., ICLR 2024).

Clone separately: git clone https://github.com/skywalker023/confaide.git data/confaide
"""

import re
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class Tier1Item:
    """Tier 1: Information sensitivity rating."""
    prompt: str
    information: str  # extracted information type
    label: float       # human sensitivity rating (1-4 scale)


@dataclass
class Tier2Item:
    """Tier 2: Information flow expectation."""
    prompt: str
    information: str
    actor: str
    purpose: str
    label: float       # human agreement rating (-100 to 100)
    variant: str       # "2a" or "2b"


@dataclass
class Tier3Item:
    """Tier 3: Multi-party information flow control."""
    story: str
    question: str       # "What should X say?"
    topic: str          # e.g., "work/academic cheating"
    about: str          # person the secret is about
    questionee: str     # person who knows the secret
    questioner: str     # person asking
    secret_topic: str   # the specific secret
    aware_relation: str
    oblivious_relation: str
    reveal_reason: str
    control_question: str  # "Is it ok for X to share..."


@dataclass
class Tier4Item:
    """Tier 4: Meeting transcript summarization."""
    transcript: str
    question: str
    question_type: str   # "meeting-summary" or "action-item"
    private_info: str
    public_info: str


class ConfaideLoader:
    """Load and iterate over CONFAIDE benchmark tiers."""

    def __init__(self, confaide_root: str = "data/confaide"):
        self.root = Path(confaide_root)
        self.benchmark_dir = self.root / "benchmark"
        if not self.benchmark_dir.exists():
            raise FileNotFoundError(
                f"CONFAIDE benchmark not found at {self.benchmark_dir}. "
                "Please clone it: git clone https://github.com/skywalker023/confaide.git data/confaide"
            )

    def load_tier1(self) -> list[Tier1Item]:
        """Load Tier 1: information sensitivity classification (10 items)."""
        prompts_path = self.benchmark_dir / "tier_1.txt"
        labels_path = self.benchmark_dir / "tier_1_labels.txt"

        prompts = self._read_lines(prompts_path)
        labels = [float(x) for x in self._read_lines(labels_path)]

        items = []
        for prompt, label in zip(prompts, labels):
            # Extract the information type after "Information: "
            match = re.search(r"Information:\s*(.+)$", prompt)
            info = match.group(1).strip() if match else prompt
            items.append(Tier1Item(prompt=prompt, information=info, label=label))
        return items

    def load_tier2(self, variant: str = "both") -> list[Tier2Item]:
        """
        Load Tier 2: information flow expectation (98 items per variant).
        variant: "2a", "2b", or "both"
        """
        labels = [float(x) for x in self._read_lines(self.benchmark_dir / "tier_2_labels.txt")]

        items = []
        variants = []
        if variant in ("2a", "both"):
            variants.append("2a")
        if variant in ("2b", "both"):
            variants.append("2b")

        for v in variants:
            prompts = self._read_lines(self.benchmark_dir / f"tier_{v}.txt")
            for i, prompt in enumerate(prompts):
                # Parse: "Information about X is collected by Y in order to Z."
                info, actor, purpose = "", "", ""
                flow_match = re.search(
                    r"Information about (.+?) is collected by (.+?) in order to (.+?)$",
                    prompt,
                )
                if flow_match:
                    info = flow_match.group(1).strip()
                    actor = flow_match.group(2).strip()
                    purpose = flow_match.group(3).strip().rstrip(".")

                label = labels[i] if i < len(labels) else 0.0
                items.append(Tier2Item(
                    prompt=prompt,
                    information=info,
                    actor=actor,
                    purpose=purpose,
                    label=label,
                    variant=v,
                ))
        return items

    def load_tier3(self) -> list[Tier3Item]:
        """Load Tier 3: multi-party privacy scenarios."""
        text = (self.benchmark_dir / "tier_3.txt").read_text(encoding="utf-8")
        control_lines = self._read_lines(self.benchmark_dir / "tier_3_control.txt")

        pattern = re.compile(
            r"<BEGIN><([^>]+)>\s*\n(.*?)\n<END><([^>]+)><([^>]+)>",
            re.DOTALL,
        )

        items = []
        for i, match in enumerate(pattern.finditer(text)):
            header = match.group(1)
            meta_str = match.group(4)

            header_parts = [p.strip() for p in header.split(",")]
            topic = header_parts[0] if len(header_parts) > 0 else ""
            aware_rel = header_parts[1] if len(header_parts) > 1 else ""
            obliv_rel = header_parts[2] if len(header_parts) > 2 else ""
            reveal_reason = ",".join(header_parts[3:]) if len(header_parts) > 3 else ""

            meta = {}
            for pair in meta_str.split(","):
                if ":" in pair:
                    key, val = pair.split(":", 1)
                    meta[key.strip()] = val.strip()

            story_text = match.group(2).strip()
            sentences = story_text.split("\n")
            story_parts = []
            question = ""
            for s in sentences:
                s = s.strip()
                if s:
                    if "What should" in s and s.endswith("?"):
                        question = s
                    else:
                        story_parts.append(s)
            story = "\n\n".join(story_parts)

            control_q = control_lines[i] if i < len(control_lines) else ""

            items.append(Tier3Item(
                story=story,
                question=question,
                topic=topic,
                about=meta.get("About", ""),
                questionee=meta.get("Questionee", ""),
                questioner=meta.get("Questioner", ""),
                secret_topic=meta.get("Topic", ""),
                aware_relation=aware_rel,
                oblivious_relation=obliv_rel,
                reveal_reason=reveal_reason,
                control_question=control_q,
            ))
        return items

    def load_tier4(self) -> list[Tier4Item]:
        """Load Tier 4: meeting transcript summarization."""
        text = (self.benchmark_dir / "tier_4.txt").read_text(encoding="utf-8")

        pattern = re.compile(
            r"<([^>]+),\s*([^>]+)>\s*\n(.*?)\n<([^>]+),\s*([^>]+)>",
            re.DOTALL,
        )

        items = []
        for match in pattern.finditer(text):
            private_info = match.group(1).strip()
            public_info = match.group(2).strip()
            body = match.group(3).strip()

            parts = body.split("\nNow generate ")
            if len(parts) >= 2:
                transcript = parts[0].strip()
                for task_part in parts[1:]:
                    question = "Now generate " + task_part.strip()
                    q_type = "meeting-summary" if "summary" in question.lower() else "action-item"
                    items.append(Tier4Item(
                        transcript=transcript,
                        question=question,
                        question_type=q_type,
                        private_info=private_info,
                        public_info=public_info,
                    ))
            else:
                items.append(Tier4Item(
                    transcript=body,
                    question="",
                    question_type="unknown",
                    private_info=private_info,
                    public_info=public_info,
                ))
        return items

    def load_all(self) -> dict[str, list]:
        """Load all tiers."""
        return {
            "tier1": self.load_tier1(),
            "tier2": self.load_tier2(),
            "tier3": self.load_tier3(),
            "tier4": self.load_tier4(),
        }

    def summary(self) -> dict[str, int]:
        """Return number of items per tier."""
        all_data = self.load_all()
        return {k: len(v) for k, v in all_data.items()}

    @staticmethod
    def _read_lines(path: Path) -> list[str]:
        """Read non-empty lines from a text file."""
        return [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
