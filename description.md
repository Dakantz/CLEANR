# GutBrain IE Challenge @ CLEF 2025: {{system_id}}

`Benedikt Kantz, Peter Walder, Stefan Lengauer, Tobias Schreck`
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


