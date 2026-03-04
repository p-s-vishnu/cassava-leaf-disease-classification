# I Trained a Model. What is Next?

> **Author:** Vladimir Iglovikov — Co-creator of albumentations.ai, PhD in Physics, Kaggle Grandmaster.
> **Published:** August 28, 2020 · 11 min read
> **Source:** [ternaus.blog](https://ternaus.blog/tutorial/2020/08/28/Trained-model-what-is-next.html) · [Archived](https://web.archive.org/web/20241103041614/https://ternaus.blog/tutorial/2020/08/28/Trained-model-what-is-next.html)

---

## Introduction

Participating in ML competitions builds machine learning muscles — but what happens after the model is trained? Code ends up in private repos, weights scattered across hard drives, and eventually everything gets deleted.

The same pattern exists in academia: a student trains a model, writes a paper, and once it's accepted, the pipeline is abandoned.

This guide covers small, actionable steps you can take after every ML challenge. These steps will:

- Boost technical knowledge
- Build a personal brand
- Improve career opportunities
- Make the world a better place

> **Reference repository used throughout:** [github.com/ternaus/retinaface](https://github.com/ternaus/retinaface)

---

## Table of Contents

| Step | Time | Action |
|------|------|--------|
| I | +5 min | Release code to a public GitHub repository |
| II | +20 min | Improve code readability |
| III | +20 min | Create a good README |
| IV | +20 min | Make it easy to use your trained model |
| V | +20 min | Make a library |
| VI | +20 min | Create a Google Colab notebook |
| VII | +20 min | Create a Web App |
| VIII | +4 hours | Write a blog post |
| IX | Days | Write a paper |

---

## I. +5 min: Release Code to a Public GitHub Repository

Most likely your code is already on GitHub — but in a **private** repo.

> The most common obstacle: people assume all public code must be perfect, and that they will be judged if it's not.

**In reality: no one cares. Just release it as-is.**

Making code public is an important psychological step. All subsequent steps depend on this one.

---

## II. +20 min: Improve Readability

You can improve the readability of your Python code by adding **syntax formatters** and **checkers**. Think of fixing syntax as basic hygiene — like brushing your teeth, but for code.

### Step 1: Configuration Files

Add these files to the root of your repository:

- `setup.cfg` — configuration for `flake8` and `mypy`
- `pyproject.toml` — configuration for `black`

### Step 2: Install Requirements

```bash
pip install black flake8 mypy
```

### Step 3: black — Auto Formatter

There are countless ways to format code. `black` enforces a consistent, pre-defined set of rules automatically.

```bash
black .
```

### Step 4: flake8 — Linter

Checks for syntax issues without modifying the code.

```bash
flake8
```

Fix whatever it outputs.

### Step 5: mypy — Type Checker

Python doesn't enforce static typing, but adding type hints improves readability and allows `mypy` to catch bugs.

```python
class MyModel(nn.Module):
    def forward(x: torch.Tensor) -> torch.Tensor:
        return self.final(x)
```

```bash
mypy .
```

### Step 6: Pre-commit Hook

Running `black`, `flake8`, and `mypy` manually is tedious. Use a **pre-commit hook** to enforce checks automatically on every commit.

```bash
pip install pre-commit
pre-commit install
```

> The key difference vs. running manually: pre-commit **forces** fixes, it doesn't just beg for them. No willpower wasted.

### Step 7: GitHub Actions — CI Pipeline

Add a second line of defense. Run checks on every pull request:

```yaml
- name: Install dependencies
  run: |
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    pip install black flake8 mypy

- name: Run black
  run: black --check .

- name: Run flake8
  run: flake8

- name: Run Mypy
  run: mypy retinaface
```

> **Tip:** Stop pushing directly to `master`. Create a branch, commit, open a pull request, merge. This is industry standard.

---

## III. +20 min: Create a Good README

A good README serves two purposes:

1. **For yourself** — you'll return to this code and won't remember what it does.
2. **For others** — the README is your selling point. Without it, people won't use your work.

### Bare Minimum for an ML Repository

- [ ] A visual/image that shows the task and your solution — no words needed
- [ ] Where to put the data
- [ ] How to start training
- [ ] How to perform inference

> If you need hundreds of words to explain how to run inference, that's a **red flag** — refactor your code to be more user-friendly.

The README you write here will be reused when building a library and a web app.

---

## IV. +20 min: Make It Easy to Use Your Trained Model

### The Problem

Typical model loading requires weights on disk and knowing where they are:

```python
model = MyFancyModel()
state_dict = torch.load(<path_to_weights>)
model.load_state_dict(state_dict)
```

### The Better Way

Leverage `torch.utils.model_zoo.load_url` — weights are downloaded automatically if not cached locally:

```python
from retinaface.pre_trained_models import get_model

model = get_model("resnet50_2020-07-20", max_size=2048)
```

This is exactly the pattern used by `torchvision` and `timm`.

### Step 1: Host Model Weights

Use **GitHub Releases** to host your weights — up to **2 GB per file**, free, no AWS/GCP required.

### Step 2: Write a Model Loader Function

```python
# retinaface/pre_trained_models.py
from collections import namedtuple
from torch.utils import model_zoo
from retinaface.predict_single import Model

ModelEntry = namedtuple("ModelEntry", ["url", "model"])

models = {
    "resnet50_2020-07-20": ModelEntry(
        url="https://github.com/ternaus/retinaface/releases/download/0.01/retinaface_resnet50_2020-07-20-f168fae3c.zip",
        model=Model,
    )
}

def get_model(model_name: str, max_size: int, device: str = "cpu") -> Model:
    entry = models[model_name]
    m = entry.model(max_size=max_size, device=device)
    state_dict = model_zoo.load_url(entry.url, progress=True, map_location="cpu")
    m.load_state_dict(state_dict)
    return m
```

---

## V. +20 min: Make a Library

The goal: allow users to run predictions **without** cloning your repo.

### Step 1: Pin Dependencies

```bash
pip freeze > requirements.txt
```

### Step 2: Restructure the Repo

Create a main package folder mirroring the repo name:

```
retinaface/
├── retinaface/
│   ├── __init__.py
│   ├── pre_trained_models.py
│   └── predict_single.py
├── tests/
├── notebooks/
├── setup.py
├── setup.cfg
└── README.md
```

### Step 3: Publish to PyPI

```bash
python setup.py sdist
python setup.py sdist upload
```

Now anyone can install your model with:

```bash
pip install retinaface-pytorch
```

> PyPI uses your README to present the project on its package page.

---

## VI. +20 min: Create a Google Colab Notebook

With `pip install` and auto-downloading weights in place, you can create a Google Colab notebook where users need only a **browser** to experiment with your model.

```python
!pip install retinaface-pytorch

from retinaface.pre_trained_models import get_model
model = get_model("resnet50_2020-07-20", max_size=2048)
```

> Add a "Open in Colab" badge to your README.

---

## VII. +20 min: Create a Web App

Building a simple demo web app with **Streamlit** is far easier than most data scientists assume.

### Deploy to Heroku

```bash
heroku login
heroku create
git push heroku master
```

---

## VIII. +4 Hours: Write a Blog Post

> "One man's trash is another man's treasure."

Many people underestimate their own knowledge. If you know how to do something, that doesn't mean everyone does.

### For ML, cover:
- What was the problem?
- How did you solve it?

Sharing knowledge in blog posts and at meetups attracts recruiters and hiring managers. It builds your brand as an expert.

---

## IX. Days: Write a Paper

Even a non-breakthrough paper has value to the community. Writing in academic format is a learnable skill — or you can collaborate with someone who has it.

> "We assumed that no one would be interested in [TernausNet]. It is my most cited work."

### The Full Package

Your paper doesn't stand alone. It comes bundled with:

- A public GitHub repo with clean code and a good README
- A pip-installable library
- A Google Colab notebook for instant experimentation
- A web app for non-technical audiences
- A blog post in plain language

Together, this package demonstrates **ownership** and **communication** — both critical for career growth.

---

## Summary

```
+5 min   -> Make GitHub repo public
+20 min  -> Add black, flake8, mypy, pre-commit, GitHub Actions
+20 min  -> Write a proper README
+20 min  -> Auto-download weights via torch.hub / model_zoo
+20 min  -> Publish to PyPI
+20 min  -> Google Colab notebook
+20 min  -> Streamlit + Heroku web demo
+4 hrs   -> Blog post
+days    -> Academic paper
```

---

*Original post by [Vladimir Iglovikov](https://ternaus.blog) · August 28, 2020*
