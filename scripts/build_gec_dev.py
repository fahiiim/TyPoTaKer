"""Build a deterministic held-out GEC development suite.

The suite is deliberately separate from generated training data. Each error
case has an equivalent clean safety case, plus additional clean-only cases.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def row(name: str, source: str, target: str, category: str, clean: bool = False) -> dict[str, object]:
    return {
        "name": name,
        "source": source,
        "target": target,
        "category": category,
        "clean": clean,
    }


TYPO_CASES = [
    ("typo-the", "This is teh final document.", "This is the final document."),
    ("typo-receive", "Please recieve the package today.", "Please receive the package today."),
    ("typo-separate", "Keep the files in seperate folders.", "Keep the files in separate folders."),
    ("typo-address", "Confirm the delivery adress first.", "Confirm the delivery address first."),
    ("typo-definitely", "I will definately attend the meeting.", "I will definitely attend the meeting."),
    ("typo-occurred", "The error occured after the update.", "The error occurred after the update."),
    ("typo-environment", "Use a clean enviroment for testing.", "Use a clean environment for testing."),
    ("typo-weird", "That is a wierd result.", "That is a weird result."),
    ("typo-because", "The task failed becuase the port was closed.", "The task failed because the port was closed."),
    ("typo-tomorrow", "I will send the report tomorow.", "I will send the report tomorrow."),
    ("typo-their", "Thier configuration is incomplete.", "Their configuration is incomplete."),
    ("typo-until", "Wait untill the process finishes.", "Wait until the process finishes."),
    ("typo-coming", "The release is comming soon.", "The release is coming soon."),
    ("typo-government", "The goverment published the report.", "The government published the report."),
    ("typo-argument", "They started an arguement yesterday.", "They started an argument yesterday."),
    ("typo-successful", "The migration was succesful.", "The migration was successful."),
    ("typo-beginning", "Read the begining of the guide.", "Read the beginning of the guide."),
    ("typo-necessary", "A backup is neccessary here.", "A backup is necessary here."),
    ("typo-calendar", "Add the event to your calender.", "Add the event to your calendar."),
    ("typo-embarrass", "Do not embarass the new user.", "Do not embarrass the new user."),
    ("typo-referred", "She refered me to the manual.", "She referred me to the manual."),
    ("typo-accommodate", "The room can accomodate ten people.", "The room can accommodate ten people."),
    ("typo-publicly", "The company announced it publically.", "The company announced it publicly."),
    ("typo-maintenance", "Schedule regular maintainance checks.", "Schedule regular maintenance checks."),
    ("typo-available", "The new version is avaliable now.", "The new version is available now."),
    ("typo-immediately", "Restart the service imediately.", "Restart the service immediately."),
    ("typo-decision", "That was a difficult descision.", "That was a difficult decision."),
    ("typo-knowledge", "Share your knowlege with the team.", "Share your knowledge with the team."),
    ("typo-independent", "The test must be independant.", "The test must be independent."),
    ("typo-privilege", "This action needs admin privilage.", "This action needs admin privilege."),
]


GRAMMAR_CASES = [
    ("grammar-have", "I has a question.", "I have a question."),
    ("grammar-goes", "She go to work by train.", "She goes to work by train."),
    ("grammar-are", "They is ready to leave.", "They are ready to leave."),
    ("grammar-were", "We was late for class.", "We were late for class."),
    ("grammar-does", "He do not know the answer.", "He does not know the answer."),
    ("grammar-has", "The server have two ports.", "The server has two ports."),
    ("grammar-was", "The results was unexpected yesterday.", "The results were unexpected yesterday."),
    ("grammar-agree", "I am agree with this plan.", "I agree with this plan."),
    ("grammar-an", "She bought a umbrella.", "She bought an umbrella."),
    ("grammar-a", "He is an university student.", "He is a university student."),
    ("grammar-listens", "The server listen on port 8000.", "The server listens on port 8000."),
    ("grammar-needs", "This file need a new name.", "This file needs a new name."),
    ("grammar-works", "My laptop work without problems.", "My laptop works without problems."),
    ("grammar-do", "The students does their homework.", "The students do their homework."),
    ("grammar-people", "There is many people outside.", "There are many people outside."),
    ("grammar-child", "Each child have a notebook.", "Each child has a notebook."),
    ("grammar-try", "She try to call every day.", "She tries to call every day."),
    ("grammar-watches", "He watch the logs carefully.", "He watches the logs carefully."),
    ("grammar-better", "This version is more better.", "This version is better."),
    ("grammar-did", "They did not went home.", "They did not go home."),
    ("grammar-at", "We met in Monday morning.", "We met on Monday morning."),
    ("grammar-since", "I have lived here since five years.", "I have lived here for five years."),
    ("grammar-explain", "Can you explain me the rule?", "Can you explain the rule to me?"),
    ("grammar-discuss", "We discussed about the issue.", "We discussed the issue."),
    ("grammar-depend", "It depends of the input.", "It depends on the input."),
    ("grammar-information", "She gave me many informations.", "She gave me much information."),
    ("grammar-advice", "He gave me a good advices.", "He gave me good advice."),
    ("grammar-everyone", "Everyone are invited tonight.", "Everyone is invited tonight."),
    ("grammar-one", "One of the files are missing.", "One of the files is missing."),
    ("grammar-much", "How many water do you need?", "How much water do you need?"),
]


MECHANICS_CASES = [
    ("mechanics-capital-i", "i need help with the installation.", "I need help with the installation."),
    ("mechanics-capital-name", "fahim sent the message.", "Fahim sent the message."),
    ("mechanics-capital-start", "the process completed successfully.", "The process completed successfully."),
    ("mechanics-dont", "I dont understand this warning.", "I don't understand this warning."),
    ("mechanics-cant", "We cant deploy this build.", "We can't deploy this build."),
    ("mechanics-isnt", "The service isnt available yet.", "The service isn't available yet."),
    ("mechanics-youre", "Youre ready to begin.", "You're ready to begin."),
    ("mechanics-theyre", "Theyre reviewing the changes.", "They're reviewing the changes."),
    ("mechanics-wont", "The script wont start.", "The script won't start."),
    ("mechanics-period", "Please save your work", "Please save your work."),
    ("mechanics-question", "Where is the configuration file", "Where is the configuration file?"),
    ("mechanics-comma", "After lunch we will test again.", "After lunch, we will test again."),
    ("mechanics-space-period", "The build is complete .", "The build is complete."),
    ("mechanics-space-comma", "First , open the terminal.", "First, open the terminal."),
    ("mechanics-double-space", "The model is  ready now.", "The model is ready now."),
    ("mechanics-its", "Its time to restart the service.", "It's time to restart the service."),
    ("mechanics-im", "Im checking the output.", "I'm checking the output."),
    ("mechanics-ive", "Ive already read the guide.", "I've already read the guide."),
    ("mechanics-weve", "Weve fixed the issue.", "We've fixed the issue."),
    ("mechanics-didnt", "The test didnt pass.", "The test didn't pass."),
]


CLEAN_ONLY = [
    "The database backup completed without errors.",
    "Please review the pull request before merging it.",
    "The application starts on the configured port.",
    "A clear error message helps users recover quickly.",
    "The model should preserve correct text exactly.",
    "Our test suite runs before every release.",
    "The new configuration file uses valid JSON.",
    "She wrote detailed notes during the meeting.",
    "The package manager installed the dependency.",
    "You can restart the service after the update.",
    "The dashboard displays the latest metrics.",
    "Please keep your access token private.",
    "The customer received the confirmation email.",
    "We will investigate the issue tomorrow morning.",
    "The command completed in less than one second.",
    "This document explains the deployment process.",
    "The release includes several security fixes.",
    "Every contributor should read the style guide.",
    "The logs contain enough context for debugging.",
    "I updated the dependency to the latest version.",
    "The network connection is stable today.",
    "They tested the feature on multiple devices.",
    "The response contains a valid status code.",
    "We should measure latency before optimization.",
    "The source file uses UTF-8 encoding.",
    "Your changes are ready for code review.",
    "The editor highlights syntax errors automatically.",
    "A small batch can limit GPU utilization.",
    "The tokenizer converts text into token IDs.",
    "Please close unused applications before training.",
    "The checkpoint stores the model and optimizer state.",
    "This experiment has a reproducible random seed.",
    "The evaluation set must remain separate from training.",
    "The graph shows a steady decline in loss.",
    "The team documented the final decision clearly.",
    "A reliable system handles invalid input gracefully.",
    "The model runs on the available CUDA device.",
    "The command line tool accepts a text argument.",
    "The report includes both successes and limitations.",
    "We will compare the models using identical settings.",
]


def build_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for category, cases in (
        ("typo", TYPO_CASES),
        ("grammar", GRAMMAR_CASES),
        ("mechanics", MECHANICS_CASES),
    ):
        for name, source, target in cases:
            rows.append(row(name, source, target, category))
            rows.append(row(f"clean-{name}", target, target, "clean", clean=True))
    for i, text in enumerate(CLEAN_ONLY, start=1):
        rows.append(row(f"clean-only-{i:02d}", text, text, "clean", clean=True))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the held-out GEC development suite")
    parser.add_argument("--out", default="data/eval/gec_dev.jsonl")
    args = parser.parse_args()

    rows = build_rows()
    if len(rows) != 200:
        raise RuntimeError(f"expected 200 development cases, got {len(rows)}")
    names = [str(item["name"]) for item in rows]
    if len(names) != len(set(names)):
        raise RuntimeError("development case names must be unique")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for item in rows:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    counts: dict[str, int] = {}
    for item in rows:
        category = str(item["category"])
        counts[category] = counts.get(category, 0) + 1
    print(f"wrote {len(rows)} held-out GEC development cases -> {out}")
    print("category_counts=" + json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
