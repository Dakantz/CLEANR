# GutBrain IE Challenge @ CLEF 2025: {{system_id}}

* Team ID: {{team_id}}
* TaskID: {{task_id}}
* RunID: {{run_id}}
* Run Flags
{{#flags}}
  - {{.}}
{{/flags}}
* GitHub: https://github.com/Dakantz/CLEANR
## Our appraoch
* Use a RAG approach to prompt a LM to return the relations
  - fetch similar articles from VectorDB to give good examples (if the run ID contains `rag`)
  - reorder the RAG data to improve the handling of the model, i.e. put Gold annotations before Silver (if the run ID contains `reorder`)
  - finetune the Hermes model on the train data combinations, with text+annotation pairs (if the run ID contains `lora`)
* We also use different models:
  - `NousResearch/Hermes-3-Llama-3.2-3B` + a finetuned LoRA-version
  - `NousResearch/Hermes-3-Llama-3.1-8B`
  - `gpt-4o-mini-2024-07-18`
* Merged with the Graphwise team, strategy based on run ID (either intersection or union):
  - Type of training applied. Finetuning `microsoft/BiomedNLP-BiomedELECTRA-base-uncased-abstract` on task T61 after that the model is further finetuned on task T623.
  - Pre-processing methods. The one provided in the baseline repo
  - Training data used. The one provided by the competition organizers plus data annotated by the gliner model provided as baseline.
  - Relevant details of the run. The model is finetuned on the training data for 100 epochs, train_batch_size 2 , gradient_accumulation_steps 2 , learning_rate 5e-5 , max_grad_norm 1.0 , warmup_ratio 0.06 