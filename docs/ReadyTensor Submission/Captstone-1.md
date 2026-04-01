Project Objectives
In this project, you'll demonstrate your ability to fine-tune an open-weights LLM end-to-end by:

Preparing a dataset for instruction fine-tuning
Establishing a baseline by evaluating the base model
Fine-tuning the model using LoRA or QLoRA (parameter-efficient methods)
Evaluating the results to show improvement and check for catastrophic forgetting
Publishing your model with proper documentation
You'll work with a small to medium-sized model (e.g., Mistral 7B, Phi-3 Mini, Qwen-1.5B, or Llama 3.2 1B-3B) and adapt it for a specific task of your choice.

Key Focus Areas:

Practical, reproducible workflows (not theoretical exercises)
Real evaluation that demonstrates actual improvement
Professional documentation and publishing practices
Cost-effective training (single GPU is sufficient; multi-GPU is optional)
What You'll Build
You'll build and document a fine-tuned model that adapts an open-weights base LLM for a specific task or dataset.

Essential Requirements (Must Complete)
These core components demonstrate the fundamental skills you've learned:

1. Dataset Selection and Preparation
Choose a publicly available dataset from Hugging Face Datasets (or similar platform)
Format your dataset appropriately for instruction fine-tuning (instruction-response pairs or chat format)
Split into train/validation sets
Document your dataset choice and any preprocessing steps
Note: You don't need to collect or create your own dataset — using existing public datasets is perfectly acceptable
2. Baseline Evaluation
Evaluate your base model (before fine-tuning) on your chosen task
Use appropriate metrics for your task (e.g., ROUGE for summarization, accuracy for classification, exact match for structured outputs)
Document baseline performance — this becomes your reference point
3. Fine-Tuning Implementation
Fine-tune using LoRA or QLoRA (parameter-efficient fine-tuning)
Use Hugging Face Transformers with PEFT, or Axolotl (your choice)
Apply 4-bit quantization (QLoRA) if needed to fit your model on available hardware
Document your training configuration (LoRA rank, learning rate, batch size, etc.)
Show training progress (loss curves, validation metrics)
4. Post-Fine-Tuning Evaluation
Evaluate your fine-tuned model using the same metrics as your baseline
Compare before/after performance to demonstrate improvement
Include at least one general benchmark (e.g., MMLU subset, HellaSwag, or GSM8K) to check for catastrophic forgetting
Document all evaluation results clearly
5. Experiment Tracking
Log at least one training run using Weights & Biases (or similar tool)
Track hyperparameters, training metrics, and evaluation results
Include a link to your W&B project in your submission
6. Model Publishing
Publish your fine-tuned model (or LoRA adapters) to Hugging Face Hub
Include a complete model card (README.md) with:
Model description and intended use
Training data and procedure
Evaluation results (baseline vs fine-tuned)
Limitations and known issues
How to load and use the model
7. Reproducible Code
Provide clean, well-documented code (notebooks or scripts)
Include a requirements.txt or environment setup instructions
Ensure someone else can reproduce your results by following your code
Optional Enhancements (Bonus Points)
These demonstrate advanced skills but are not required:

Multi-GPU Training: If you have access to multiple GPUs, demonstrate DDP, FSDP, or DeepSpeed ZeRO
Hyperparameter Tuning: Run multiple experiments comparing different LoRA ranks, learning rates, or target modules
Advanced Evaluation: Build a custom evaluation suite with operational checks (length constraints, format validation, etc.)
Model Merging: Merge LoRA adapters into base model and publish merged version
Additional Benchmarks: Run multiple general benchmarks to thoroughly assess capability retention
Example Project Ideas
Not sure what to fine-tune your model for? Here are a few directions to help you choose a dataset and define your project goal.

All of these can be done using existing public datasets available on 
Hugging Face Datasets
.
You don’t need to collect or clean your own data — simply pick a relevant dataset and focus on fine-tuning, evaluation, and optimization.

Domain-Specific Instruction Tuning
Fine-tune your model to handle domain-specific queries using a dataset specialized in a particular field such as finance, healthcare, or legal text.

Example: fine-tune on financial QA or stock report datasets so the model can answer questions about companies or filings in a concise, factual way.
Objective: improve domain fluency and response accuracy compared to the base model.
Technical Assistant for Developers
Use a dataset of technical commands or documentation (e.g., Docker, SQL, Git, or Linux) to train your model to respond to natural-language queries.

Example: fine-tune a model that converts prompts like “create a new Docker container for Redis” into the right command syntax.
Objective: demonstrate task-specific instruction tuning and precise text generation.
Reasoning and Problem Solving
Work with benchmark datasets such as GSM8K (grade-school math problems) or ARC (AI2 Reasoning Challenge) to improve reasoning or step-by-step problem-solving ability.

Example: fine-tune your model on a subset of GSM8K to teach it how to reason through arithmetic problems.
Objective: evaluate the model’s reasoning capability before and after fine-tuning.
Explore Your Own Idea
You’re not limited to these examples — explore any dataset or task that interests you.
Just ensure the dataset is publicly available, appropriately licensed, and small enough to fine-tune efficiently.

If you’re unsure where to start, browse 
Hugging Face Datasets
 for inspiration — many datasets include sample scripts and documentation that make setup easy.

To be included in a given month’s review cycle, make sure to submit your project by one of the following dates:

 🔴 January 05, 2026,11:59 PM UTC
 🔴 February 02, 2026, 11:59 PM UTC
 🔴 March 02, 2026, 11:59 PM UTC
 🔴 April 06, 2026, 11:59 PM UTC
 🔴 May 04, 2026, 11:59 PM UTC
If you miss a listed date, your project will simply roll over to the next month’s review.
Reviews typically take about two weeks, during which you’ll receive feedback and, if needed, an opportunity to make improvements before final evaluation.

Plan ahead so you can complete your submission comfortably within your preferred review window.

Submission Checklist
To complete this project, submit the following deliverables:

1. Project Publication on Ready Tensor
Create a technical publication that documents your project:

Required Content:

Objective: What task are you fine-tuning for? Why did you choose this task?
Dataset: Which dataset did you use? How did you prepare it?
Methodology:
Base model selection
Fine-tuning approach (LoRA/QLoRA configuration)
Training setup (hardware, framework, hyperparameters)
Results:
Baseline vs fine-tuned performance comparison
Training curves (loss over time)
Evaluation metrics (task-specific + at least one general benchmark)
Discussion: What worked well? What challenges did you face?
Visual Elements:

Charts showing training loss curves
Tables comparing baseline vs fine-tuned metrics
Example inputs/outputs demonstrating improvement
The publication should meet at least 70% of the Technical Evaluation Rubric for technical publications.

📄 
Publication Evaluation Rubric

2. GitHub Repository
Submit a well-organized repository with:

Required Files:

README.md: Project overview, setup instructions, how to reproduce
Training code: Scripts or notebooks for dataset preparation and fine-tuning
Evaluation code: Scripts for baseline and post-fine-tuning evaluation
requirements.txt: All Python dependencies with versions
Model card: Documentation of your model (can be in README or separate file)
Code Quality:

Code should be clean, commented, and reproducible
Someone else should be able to run your code and get similar results
Document any assumptions or environment-specific configurations
The repository should meet 70% of the "Essential" level in the repository evaluation rubric.

📄 
Repository Evaluation Rubric

3. Published Model on Hugging Face Hub
Required:

Model (or LoRA adapters) published to Hugging Face Hub
Complete model card (README.md) with all required sections
Evaluation results included in model card or as separate file
Model Card Must Include:

Model description and intended use
Training data information
Training procedure and hyperparameters
Evaluation results (baseline vs fine-tuned)
Limitations and known issues
Code example for loading and using the model
4. Experiment Tracking (Weights & Biases)
Required:

At least one training run logged to W&B
Link to your W&B project included in your GitHub README or publication
Should Include:

Hyperparameters
Training metrics (loss, learning rate schedule)
Evaluation metrics
System information (GPU type, memory usage)
What You'll Earn
Successfully completing this project earns you the LLM Fine-Tuning Specialist credential — recognizing your ability to fine-tune, evaluate, and publish large language models using parameter-efficient methods.

If you've also completed Module 2 and earned the LLM Deployment Engineer credential, you'll be awarded the LLM Engineering & Deployment Certification, representing full completion of the program.

Your Next Step
Once you’ve submitted your project, it will be reviewed by the evaluation team.
If it meets the certification standards, you’ll receive your credential — and if you’ve completed both modules, your full program certificate as well.

This marks your official recognition as a certified LLM Engineer, capable of taking models from fine-tuning to production.